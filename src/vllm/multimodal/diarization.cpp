// diarization.cpp — diarization seam wrapping parakeet.cpp's C-API
//
// parakeet.cpp's offline diarization API returns a JSON string (char*),
// not a struct. We parse it here to extract speaker segments.
#include "vllm/multimodal/diarization.h"

#ifdef VLLM_WITH_DIARIZATION
#include "vllm/multimodal/parakeet_transcription.h"
#include <nlohmann/json.hpp>
#endif

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef VLLM_WITH_DIARIZATION

namespace vllm::multimodal {

// --- WAV reading helper ---
// Reads a 16-bit PCM mono WAV into float32 samples.
static std::vector<float> ReadWavPcm16Mono(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open WAV: " + path);
    char hdr[44];
    if (std::fread(hdr, 1, 44, f) != 44) {
        std::fclose(f);
        throw std::runtime_error("WAV too short: " + path);
    }
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        std::fclose(f);
        throw std::runtime_error("not a RIFF/WAVE file: " + path);
    }
    std::fseek(f, 44, SEEK_SET);
    std::vector<int16_t> pcm16;
    int16_t sample;
    while (std::fread(&sample, 2, 1, f) == 1)
        pcm16.push_back(sample);
    std::fclose(f);
    std::vector<float> pcm(pcm16.size());
    for (size_t i = 0; i < pcm16.size(); ++i)
        pcm[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    return pcm;
}

// Parse the JSON string returned by parakeet_capi_diarize_path / _pcm.
// Format: {"segments": [{"speaker": N, "start": S, "end": E}, ...]}
static std::vector<SpeakerSegment> ParseDiarizationJson(const char* json_str) {
    std::vector<SpeakerSegment> segs;
    if (!json_str) return segs;
    auto j = nlohmann::json::parse(json_str);
    if (j.contains("segments")) {
        for (const auto& s : j["segments"]) {
            SpeakerSegment seg;
            seg.speaker = s.value("speaker", -1);
            seg.start = s.value("start", 0.0f);
            seg.end = s.value("end", 0.0f);
            segs.push_back(seg);
        }
    }
    return segs;
}

// --- Diarizer ---

Diarizer::Diarizer() = default;

Diarizer::~Diarizer() {
    if (ctx_) {
        parakeet_capi_free(ctx_);
        ctx_ = nullptr;
    }
}

std::unique_ptr<Diarizer> Diarizer::FromFile(const std::string& path) {
    auto d = std::unique_ptr<Diarizer>(new Diarizer());
    d->ctx_ = parakeet_capi_load(path.c_str());
    if (!d->ctx_) {
        throw std::runtime_error("Diarizer::FromFile: parakeet_capi_load failed: " + path);
    }
    return d;
}

std::vector<SpeakerSegment> Diarizer::Diarize(
        const float* pcm, int64_t n_samples, int sample_rate) const {
    if (!ctx_) throw std::runtime_error("Diarizer: no model loaded");

    char* json = parakeet_capi_diarize_pcm(
        ctx_, pcm, (int)n_samples, sample_rate);
    if (!json) throw std::runtime_error("Diarizer: diarize_pcm failed");

    auto segs = ParseDiarizationJson(json);
    parakeet_capi_free_string(json);
    return segs;
}

std::vector<SpeakerSegment> Diarizer::DiarizeWavFile(const std::string& path) const {
    if (!ctx_) throw std::runtime_error("Diarizer: no model loaded");

    char* json = parakeet_capi_diarize_path(ctx_, path.c_str());
    if (!json) throw std::runtime_error("Diarizer: diarize_path failed");

    auto segs = ParseDiarizationJson(json);
    parakeet_capi_free_string(json);
    return segs;
}

// --- Combined ASR + diarization ---

SpeakerAttributedASR TranscribeAndDiarize(
        const std::string& wav_path,
        const std::string& asr_dir,
        const std::string& diar_gguf) {
    SpeakerAttributedASR result;

    ParakeetTranscriber asr = ParakeetTranscriber::FromDir(asr_dir);
    auto diar = Diarizer::FromFile(diar_gguf);

    ParakeetTranscription trans = asr.TranscribeWavFile(wav_path);
    if (!trans.has_text) {
        result.has_result = false;
        return result;
    }

    auto pcm = ReadWavPcm16Mono(wav_path);

    parakeet_ctx* asr_ctx = parakeet_capi_load(asr_dir.c_str());
    if (!asr_ctx) {
        result.utterances.push_back({-1, trans.text, 0.0f, 0.0f, 0.0f});
        result.has_result = true;
        return result;
    }

    int n_sas = 0;
    parakeet_sas_result* sas = parakeet_capi_transcribe_and_diarize(
        asr_ctx, diar->ctx(), pcm.data(), (int)pcm.size(), 16000, &n_sas);

    if (sas && n_sas > 0) {
        for (int i = 0; i < n_sas; ++i) {
            SpeakerUtterance u;
            u.speaker = sas[i].speaker;
            u.text = sas[i].text ? sas[i].text : "";
            u.start = sas[i].start;
            u.end = sas[i].end;
            u.conf = sas[i].conf;
            result.utterances.push_back(u);
            if (sas[i].text) parakeet_capi_free_string(sas[i].text);
        }
        parakeet_capi_free_sas_results(sas);
        result.has_result = true;
    } else {
        result.has_result = false;
    }

    parakeet_capi_free(asr_ctx);
    return result;
}

SpeakerAttributedASR TranscribeAndDiarizePCM(
        const float* pcm, int64_t n_samples, int sample_rate,
        const std::string& asr_dir,
        const std::string& diar_gguf) {
    SpeakerAttributedASR result;

    ParakeetTranscriber asr = ParakeetTranscriber::FromDir(asr_dir);
    auto diar = Diarizer::FromFile(diar_gguf);

    ParakeetTranscription trans = asr.Transcribe(pcm, n_samples, sample_rate);
    if (!trans.has_text) {
        result.has_result = false;
        return result;
    }

    parakeet_ctx* asr_ctx = parakeet_capi_load(asr_dir.c_str());
    if (!asr_ctx) {
        result.utterances.push_back({-1, trans.text, 0.0f, 0.0f, 0.0f});
        result.has_result = true;
        return result;
    }

    int n_sas = 0;
    parakeet_sas_result* sas = parakeet_capi_transcribe_and_diarize(
        asr_ctx, diar->ctx(), pcm, (int)n_samples, sample_rate, &n_sas);

    if (sas && n_sas > 0) {
        for (int i = 0; i < n_sas; ++i) {
            SpeakerUtterance u;
            u.speaker = sas[i].speaker;
            u.text = sas[i].text ? sas[i].text : "";
            u.start = sas[i].start;
            u.end = sas[i].end;
            u.conf = sas[i].conf;
            result.utterances.push_back(u);
            if (sas[i].text) parakeet_capi_free_string(sas[i].text);
        }
        parakeet_capi_free_sas_results(sas);
        result.has_result = true;
    } else {
        result.has_result = false;
    }

    parakeet_capi_free(asr_ctx);
    return result;
}

}  // namespace vllm::multimodal

#else  // !VLLM_WITH_DIARIZATION

namespace vllm::multimodal {

std::unique_ptr<Diarizer> Diarizer::FromFile(const std::string&) {
    throw std::runtime_error("Diarizer: diarization support not compiled in");
}

std::vector<SpeakerSegment> Diarizer::Diarize(const float*, int64_t, int) const {
    throw std::runtime_error("Diarizer: diarization support not compiled in");
}

std::vector<SpeakerSegment> Diarizer::DiarizeWavFile(const std::string&) const {
    throw std::runtime_error("Diarizer: diarization support not compiled in");
}

Diarizer::Diarizer() = default;
Diarizer::~Diarizer() = default;

SpeakerAttributedASR TranscribeAndDiarize(
        const std::string&, const std::string&, const std::string&) {
    SpeakerAttributedASR r;
    r.has_result = false;
    return r;
}

SpeakerAttributedASR TranscribeAndDiarizePCM(
        const float*, int64_t, int, const std::string&, const std::string&) {
    SpeakerAttributedASR r;
    r.has_result = false;
    return r;
}

}  // namespace vllm::multimodal

#endif  // VLLM_WITH_DIARIZATION
