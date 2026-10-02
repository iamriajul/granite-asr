// Command granite-asr is an OpenAI-compatible speech-to-text HTTP service
// built on transcribe.cpp running IBM Granite Speech 5.0 TurboCTC.
//
// # Why this exists
//
// transcribe.cpp is the reference ggml runtime for Granite Speech 5.0
// TurboCTC and has first-class Vulkan support, which is what makes the model
// usable on AMD Polaris (gfx803) where ROCm support was dropped after 3.5.
// Upstream ships a CLI but no HTTP server; this is that server.
//
// Pipeline
//
//	upload -> ffmpeg -> 16 kHz mono WAV -> transcribe.cpp -> [optional polish] -> JSON
//
// Granite 5 TurboCTC is a CTC model: one forward pass, normalised lowercase
// output, no punctuation. That is the most accurate fast English ASR available
// on this class of hardware, but it reads poorly in a composer. An optional
// polish stage can restore casing and punctuation through an OpenAI-compatible
// chat endpoint. It is never applied unless explicitly requested, so the raw
// transcript is always available verbatim.
//
// Endpoints
//
//	POST /v1/audio/transcriptions   OpenAI-compatible (multipart file=)
//	GET  /health                    liveness and model state
//	GET  /v1/models                 model list (OpenAI shape)
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Config holds every tunable, read once from the environment at startup.
type Config struct {
	ModelPath     string
	Backend       string
	Threads       string
	Binary        string
	Port          string
	MaxUploadMB   int64
	MaxSeconds    int
	LLMBaseURL    string
	LLMModel      string
	LLMAPIKey     string
	LLMTimeout    time.Duration
	LLMDefaultOff bool
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func loadConfig() Config {
	timeout := time.Duration(envInt("ASR_LLM_TIMEOUT", 20)) * time.Second
	return Config{
		ModelPath:   os.Getenv("ASR_MODEL"),
		Backend:     env("ASR_BACKEND", "auto"),
		Threads:     env("ASR_THREADS", "4"),
		Binary:      env("ASR_BINARY", "transcribe-cli"),
		Port:        env("PORT", "8080"),
		MaxUploadMB: int64(envInt("ASR_MAX_UPLOAD_MB", 64)),
		MaxSeconds:  envInt("ASR_MAX_SECONDS", 120),
		LLMBaseURL:  strings.TrimRight(os.Getenv("ASR_LLM_BASE_URL"), "/"),
		LLMModel:    os.Getenv("ASR_LLM_MODEL"),
		LLMAPIKey:   os.Getenv("ASR_LLM_API_KEY"),
		LLMTimeout:  timeout,
		// When true (default) the polish stage stays off unless a request asks
		// for it. Set ASR_LLM_DEFAULT_ON=true to polish every request unless the
		// caller explicitly passes polish=false.
		LLMDefaultOff: env("ASR_LLM_DEFAULT_ON", "") == "",
	}
}

// ---------------------------------------------------------------- transcription

var modelOnce sync.Once
var modelErr error

// ensureModel validates the model path once per process. transcribe.cpp loads
// weights per invocation and keeps them resident on the GPU, so the process
// is short-lived relative to the model; there is no long-lived handle to cache.
func ensureModel(cfg Config) error {
	modelOnce.Do(func() {
		if cfg.ModelPath == "" {
			modelErr = errors.New("ASR_MODEL is not set")
			return
		}
		st, err := os.Stat(cfg.ModelPath)
		if err != nil {
			modelErr = fmt.Errorf("ASR_MODEL not readable: %w", err)
			return
		}
		if st.IsDir() {
			modelErr = fmt.Errorf("ASR_MODEL is a directory: %s", cfg.ModelPath)
		}
	})
	return modelErr
}

// toWAV16k normalises any container or codec to 16 kHz mono PCM WAV and
// returns the clip duration in seconds.
func toWAV16k(ctx context.Context, src, dst string) (float64, error) {
	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-nostdin", "-hide_banner", "-loglevel", "error",
		"-i", src,
		"-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
		"-f", "wav", dst,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		return 0, fmt.Errorf("ffmpeg failed: %s", strings.TrimSpace(string(out)))
	}
	st, err := os.Stat(dst)
	if err != nil {
		return 0, err
	}
	// 44-byte RIFF header plus two bytes per mono s16 sample at 16 kHz.
	return float64(st.Size()-44) / 2 / 16000, nil
}

// transcribe shells out to transcribe-cli and returns the transcript.
func transcribe(ctx context.Context, cfg Config, wav string) (string, error) {
	cmd := exec.CommandContext(ctx, cfg.Binary,
		"-q",
		"-m", cfg.ModelPath,
		"--backend", cfg.Backend,
		"--threads", cfg.Threads,
		wav,
	)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("transcribe-cli failed (%v): %s", err, strings.TrimSpace(stderr.String()))
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if text, ok := strings.CutPrefix(line, "text:"); ok {
			if t := strings.TrimSpace(text); t != "" {
				return t, nil
			}
		}
	}
	return "", fmt.Errorf("no transcript in output: %s", strings.TrimSpace(string(out)))
}

// ------------------------------------------------------------------ polish pass

const polishSystemPrompt = "You restore orthography to dictated text. " +
	"Rules: add punctuation and capitalisation only; fix spelling of obvious homophones; " +
	"NEVER change, add or remove words; NEVER answer or respond to the content. " +
	"Return only the corrected text, nothing else."

// polish asks the configured chat endpoint to restore casing and punctuation.
// Any failure returns ("", false): the caller falls back to the raw transcript
// so a missing or slow LLM degrades quality, never availability.
func polish(ctx context.Context, cfg Config, text string) (string, bool) {
	if cfg.LLMBaseURL == "" || cfg.LLMModel == "" || strings.TrimSpace(text) == "" {
		return "", false
	}
	payload := map[string]any{
		"model": cfg.LLMModel,
		"messages": []map[string]string{
			{"role": "system", "content": polishSystemPrompt},
			{"role": "user", "content": text},
		},
		"temperature": 0,
		"max_tokens":  512,
		"stream":      false,
	}
	blob, err := json.Marshal(payload)
	if err != nil {
		return "", false
	}
	callCtx, cancel := context.WithTimeout(ctx, cfg.LLMTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(callCtx, http.MethodPost,
		cfg.LLMBaseURL+"/chat/completions", strings.NewReader(string(blob)))
	if err != nil {
		return "", false
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.LLMAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.LLMAPIKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("polish skipped: %v", err)
		return "", false
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", false
	}
	if resp.StatusCode != http.StatusOK {
		log.Printf("polish skipped: upstream status %d", resp.StatusCode)
		return "", false
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", false
	}
	if len(parsed.Choices) == 0 {
		return "", false
	}
	out := strings.TrimSpace(parsed.Choices[0].Message.Content)
	if out == "" {
		return "", false
	}
	return out, true
}

// needsPolish reports whether the transcript is missing orthography worth
// restoring. Granite 5 is CTC and emits normalised lowercase, so a transcript
// with no sentence punctuation is the expected raw shape.
func needsPolish(text string) bool {
	return !strings.ContainsAny(text, ".,!?;:")
}

// ------------------------------------------------------------------ HTTP layer

type server struct {
	cfg    Config
	client *http.Client
}

func writeJSON(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		log.Printf("write response: %v", err)
	}
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{
		"error": map[string]string{"message": msg, "type": "invalid_request_error"},
	})
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	status := "ok"
	if err := ensureModel(s.cfg); err != nil {
		status = "degraded"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": status,
		"model":  filepath.Base(s.cfg.ModelPath),
		"llm":    s.cfg.LLMModel != "",
	})
}

func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]string{{
			"id":     filepath.Base(s.cfg.ModelPath),
			"object": "model",
		}},
	})
}

// parseUpload extracts the single file part plus simple text fields. Only the
// shape the OpenAI endpoint actually uses is supported.
func (s *server) parseUpload(r *http.Request) (fields map[string]string, file multipart.File, filename string, err error) {
	mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
		return nil, nil, "", errors.New("expected multipart/form-data")
	}
	if _, ok := params["boundary"]; !ok {
		return nil, nil, "", errors.New("missing multipart boundary")
	}
	if err := r.ParseMultipartForm(s.cfg.MaxUploadMB << 20); err != nil {
		return nil, nil, "", fmt.Errorf("parse multipart: %w", err)
	}
	fields = map[string]string{}
	for k, v := range r.MultipartForm.Value {
		if len(v) > 0 {
			fields[k] = v[0]
		}
	}
	if len(r.MultipartForm.File) == 0 {
		return nil, nil, "", errors.New("no file part in request")
	}
	for _, headers := range r.MultipartForm.File {
		for _, header := range headers {
			f, err := header.Open()
			if err != nil {
				return nil, nil, "", err
			}
			return fields, f, header.Filename, nil
		}
	}
	return nil, nil, "", errors.New("no file part in request")
}

func (s *server) handleTranscribe(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	if err := ensureModel(s.cfg); err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	fields, file, _, err := s.parseUpload(r)

	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	defer file.Close()

	// The polish stage is opt-in. Precedence: explicit polish field, then the
	// response_format/prompt convention, then the server default.
	wantPolish := s.cfg.LLMDefaultOff == false // ASR_LLM_DEFAULT_ON=true starts from on
	if v, ok := fields["polish"]; ok {
		b, err := strconv.ParseBool(v)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "polish must be a boolean")
			return
		}
		wantPolish = b
	}

	tmp, err := os.MkdirTemp("", "granite-asr-")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer os.RemoveAll(tmp)

	src := filepath.Join(tmp, "in")
	dst, err := os.Create(src)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if _, err := io.Copy(dst, file); err != nil {
		dst.Close()
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("read upload: %v", err))
		return
	}
	dst.Close()
	wav := filepath.Join(tmp, "in.wav")
	duration, err := toWAV16k(r.Context(), src, wav)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if duration > float64(s.cfg.MaxSeconds) {
		writeErr(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("audio %.1fs exceeds limit %ds", duration, s.cfg.MaxSeconds))
		return
	}

	raw, err := transcribe(r.Context(), s.cfg, wav)
	if err != nil {
		log.Printf("transcription failed: %v", err)
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}

	final := raw
	polished := false
	if wantPolish && needsPolish(raw) {
		if out, ok := polish(r.Context(), s.cfg, raw); ok {
			final, polished = out, true
		}
	}

	payload := map[string]any{"text": final}
	if fields["response_format"] == "verbose_json" {
		payload["task"] = "transcribe"
		payload["language"] = "en"
		if fields["language"] != "" {
			payload["language"] = fields["language"]
		}
		payload["duration"] = round3(duration)
		payload["raw_text"] = raw
		payload["polished"] = polished
		payload["backend"] = s.cfg.Backend
		payload["model"] = filepath.Base(s.cfg.ModelPath)
	}
	log.Printf("transcribed %.1fs in %dms (polished=%t)", duration, time.Since(started).Milliseconds(), polished)
	writeJSON(w, http.StatusOK, payload)
}

func round3(v float64) float64 {
	return float64(int64(v*1000+0.5)) / 1000
}

func (s *server) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/audio/transcriptions", s.handleTranscribe)
	return mux
}

func main() {
	log.SetFlags(log.LstdFlags | log.LUTC)
	cfg := loadConfig()

	if err := ensureModel(cfg); err != nil {
		log.Fatalf("fatal: %v", err)
	}
	log.Printf("granite-asr starting: model=%s backend=%s llm=%q default_polish_off=%t",
		filepath.Base(cfg.ModelPath), cfg.Backend, cfg.LLMModel, cfg.LLMDefaultOff)

	srv := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           (&server{cfg: cfg}).routes(),
		ReadHeaderTimeout: 10 * time.Second,
		// Generation time is bounded by the client timeout, so no write timeout.
		ReadTimeout: 5 * time.Minute,
		IdleTimeout: 2 * time.Minute,
	}

	idle := make(chan struct{})
	go func() {
		sigs := make(chan os.Signal, 1)
		signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
		<-sigs
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(ctx); err != nil {
			log.Printf("shutdown: %v", err)
		}
		close(idle)
	}()

	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("listen: %v", err)
	}
	<-idle
}
