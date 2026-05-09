import glob
import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

# ==========================================
# 1. 定義影片資料結構
# ==========================================
class VisualFeatures(BaseModel):
    lighting_contrast: str = Field(description="Lighting contrast: high/medium/low")
    color_temperature: str = Field(description="Color temperature: warm/neutral/cold")
    camera_movement: str = Field(
        description="Camera movement: Static/Dynamic/Fast-paced"
    )


class SELMetrics(BaseModel):
    self_awareness: str = Field(description="評估影片主角對自我情緒的辨識與表達深度")
    social_awareness: str = Field(description="評估影片是否展現出同理心或對多元觀點的尊重")
    responsible_decision_making: str = Field(
        description="敘事中是否包含解決問題或負責任的決策過程"
    )


class HumanisticLiteracy(BaseModel):
    critical_reflection: str = Field(
        description="影片內容引發批判性思考或自我反省的程度 (High/Medium/Low)"
    )
    value_orientation: list[str] = Field(
        description="影片傳達的核心人文價值觀，例如：包容、永續、生命關懷"
    )
    aesthetic_appreciation: str = Field(description="影像與藝術表現對人文氛圍的營造效果")


class AnalysisScores(BaseModel):
    sel_score: int = Field(ge=0, le=100, description="SEL能力值評分")
    humanistic_score: int = Field(ge=0, le=100, description="人文省思深度評分")
    narrative_impact: int = Field(ge=0, le=100, description="敘事感染力評分")

    @property
    def final_score(self) -> float:
        return (self.sel_score * 0.4) + (self.humanistic_score * 0.4) + (self.narrative_impact * 0.2)


class VideoImageAnalysis(BaseModel):
    visual_features: VisualFeatures
    audio_features: str = Field(
        description="Audio features: Inspirational/Quiet/Lively/Ambient"
    )
    editing_pace: str = Field(description="Editing pace: Slow/Moderate/Fast")
    engagement_type: str = Field(
        description="Engagement type: Educational/Social/Ritual/Labor"
    )
    human_interaction_intensity: str = Field(description="Interaction intensity: high/medium/low")
    cognitive_objects: list[str] = Field(
        description="Core cognitive objects: list 3-5 key elements in the video"
    )
    valence: float = Field(description="Valence: -1.0 to 1.0 (overall emotion)")
    arousal: float = Field(description="Arousal: -1.0 to 1.0 (dynamics and pacing)")
    narrative_arc: str = Field(description="Narrative arc: emotional change from start to end")
    reasoning: str = Field(description="Reasoning for the judgment, especially dynamic effects")
    sel_analysis: SELMetrics
    humanistic_literacy: HumanisticLiteracy
    analysis_scores: AnalysisScores


# ==========================================
# 2. 影片版 System Prompt
# ==========================================
SYSTEM_INSTRUCTION = """
You are an expert in tourism visual psychology with professional cinematography and directing skills.
Analyze dynamic social videos of international volunteers in northern Thailand.
Map visual and temporal cues to the Russell circumplex model.

Guidelines:
1. Arousal: include editing pace and camera movement in addition to contrast.
2. Valence: prioritize interpersonal interaction cues (smiles, teaching, friendly contact).
3. Temporal continuity: evaluate the full emotional flow, not a single frame.
4. Add audio_features using one label: Inspirational, Quiet, Lively, or Ambient.
5. Add editing_pace using one label: Slow, Moderate, or Fast.
6. Add engagement_type using one label: Educational, Social, Ritual, or Labor.
7. Add sel_analysis with self_awareness, social_awareness, and responsible_decision_making in Traditional Chinese.
8. Add humanistic_literacy with critical_reflection (High/Medium/Low), value_orientation (3-5 Traditional Chinese values), and aesthetic_appreciation in Traditional Chinese.
9. Add analysis_scores: sel_score, humanistic_score, and narrative_impact, each from 0 to 100.
10. Language: write all free-text fields in Traditional Chinese (Taiwan), including cognitive_objects, narrative_arc, reasoning, sel_analysis, and humanistic_literacy.
11. Keep categorical labels concise and consistent.

Output requirement: return strict JSON only.
"""

# ==========================================
# 3. 影片批次處理邏輯
# ==========================================
def analyze_videos(folder_path: str, api_key: str):
    client = genai.Client(api_key=api_key)

    video_paths = []
    for ext in ["*.mp4", "*.mov", "*.avi", "*.m4v"]:
        video_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    results = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"video_analysis_results_{stamp}.csv"
    MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro").strip()
    FALLBACK_MODEL_ID = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-pro").strip()
    RETRY_MAX = int(os.environ.get("GEMINI_RETRY_MAX", "5"))
    RETRY_BASE_SECONDS = float(os.environ.get("GEMINI_RETRY_BASE_SECONDS", "3"))
    category_zh_map = {
        "audio_features": {
            "Inspirational": "勵志",
            "Quiet": "寧靜",
            "Lively": "熱鬧",
            "Ambient": "環境音為主",
        },
        "editing_pace": {
            "Slow": "緩慢沉穩",
            "Moderate": "適中",
            "Fast": "節奏快速",
        },
        "engagement_type": {
            "Educational": "教育服務",
            "Social": "社交互動",
            "Ritual": "儀式合照",
            "Labor": "勞務實踐",
        },
        "interaction": {
            "high": "高",
            "medium": "中",
            "low": "低",
        },
        "lighting": {
            "high": "高反差",
            "medium": "中反差",
            "low": "低反差",
        },
        "color_temp": {
            "warm": "暖色",
            "neutral": "中性",
            "cold": "冷色",
        },
        "movement": {
            "Static": "固定",
            "Dynamic": "運鏡",
            "Fast-paced": "節奏快",
        },
        "critical_reflection": {
            "High": "高",
            "Medium": "中",
            "Low": "低",
        },
    }

    def to_zh(field_name: str, value):
        if value is None:
            return None
        return category_zh_map.get(field_name, {}).get(str(value), str(value))

    def generate_with_retry(model_name: str, video_file_obj):
        last_error = None
        for attempt in range(1, RETRY_MAX + 1):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=["Analyze this video's tourism destination imagery.", video_file_obj],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=VideoImageAnalysis,
                        temperature=0.2,
                    ),
                )
            except Exception as retry_error:
                last_error = retry_error
                err_text = str(retry_error)
                retriable = ("503" in err_text) or ("UNAVAILABLE" in err_text)
                if (not retriable) or attempt == RETRY_MAX:
                    raise
                wait_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(
                    f"[WARN] {model_name} busy (attempt {attempt}/{RETRY_MAX}), retry in {wait_seconds:.1f}s...",
                    end=" ",
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise last_error

    print(f"Found {len(video_paths)} videos. Start analysis...")

    for idx, path in enumerate(video_paths, start=1):
        filename = os.path.basename(path)
        print(f"({idx}/{len(video_paths)}) Processing: {filename}...", end=" ", flush=True)

        try:
            # Some SDK/transport stacks on Windows can fail with non-ASCII filenames.
            # Upload from a temporary ASCII-only filename to avoid codec errors.
            source_path = Path(path)
            temp_upload_path = source_path
            if not source_path.name.isascii():
                safe_name = f"upload_{uuid.uuid4().hex}{source_path.suffix.lower()}"
                temp_upload_path = source_path.parent / safe_name
                shutil.copy2(source_path, temp_upload_path)

            video_file = client.files.upload(file=str(temp_upload_path))

            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.name}")

            try:
                response = generate_with_retry(MODEL_ID, video_file)
            except Exception as model_error:
                err_text = str(model_error)
                if (
                    "NOT_FOUND" in err_text
                    and MODEL_ID != FALLBACK_MODEL_ID
                    and FALLBACK_MODEL_ID
                ):
                    print(
                        f"[WARN] Model '{MODEL_ID}' unavailable; fallback to '{FALLBACK_MODEL_ID}'.",
                        end=" ",
                        flush=True,
                    )
                    response = generate_with_retry(FALLBACK_MODEL_ID, video_file)
                else:
                    raise

            data = json.loads(response.text)
            sel_score = data.get("analysis_scores", {}).get("sel_score")
            humanistic_score = data.get("analysis_scores", {}).get("humanistic_score")
            narrative_impact = data.get("analysis_scores", {}).get("narrative_impact")
            final_score = None
            if all(isinstance(v, (int, float)) for v in [sel_score, humanistic_score, narrative_impact]):
                final_score = (sel_score * 0.4) + (humanistic_score * 0.4) + (narrative_impact * 0.2)
            flat_data = {
                "filename": filename,
                "audio_features": data.get("audio_features"),
                "audio_features_zh": to_zh("audio_features", data.get("audio_features")),
                "editing_pace": data.get("editing_pace"),
                "editing_pace_zh": to_zh("editing_pace", data.get("editing_pace")),
                "engagement_type": data.get("engagement_type"),
                "engagement_type_zh": to_zh("engagement_type", data.get("engagement_type")),
                "interaction": data.get("human_interaction_intensity"),
                "interaction_zh": to_zh(
                    "interaction", data.get("human_interaction_intensity")
                ),
                "objects": ", ".join(data.get("cognitive_objects", [])),
                "lighting": data.get("visual_features", {}).get("lighting_contrast"),
                "lighting_zh": to_zh(
                    "lighting", data.get("visual_features", {}).get("lighting_contrast")
                ),
                "color_temp": data.get("visual_features", {}).get("color_temperature"),
                "color_temp_zh": to_zh(
                    "color_temp", data.get("visual_features", {}).get("color_temperature")
                ),
                "movement": data.get("visual_features", {}).get("camera_movement"),
                "movement_zh": to_zh(
                    "movement", data.get("visual_features", {}).get("camera_movement")
                ),
                "valence": data.get("valence"),
                "arousal": data.get("arousal"),
                "narrative": data.get("narrative_arc"),
                "reasoning": data.get("reasoning"),
                "sel_self_awareness": data.get("sel_analysis", {}).get("self_awareness"),
                "sel_social_awareness": data.get("sel_analysis", {}).get("social_awareness"),
                "sel_responsible_decision_making": data.get("sel_analysis", {}).get("responsible_decision_making"),
                "humanistic_critical_reflection": data.get("humanistic_literacy", {}).get("critical_reflection"),
                "humanistic_critical_reflection_zh": to_zh(
                    "critical_reflection",
                    data.get("humanistic_literacy", {}).get("critical_reflection"),
                ),
                "humanistic_value_orientation": ", ".join(
                    data.get("humanistic_literacy", {}).get("value_orientation", [])
                ),
                "humanistic_aesthetic_appreciation": data.get("humanistic_literacy", {}).get("aesthetic_appreciation"),
                "sel_score": sel_score,
                "humanistic_score": humanistic_score,
                "narrative_impact": narrative_impact,
                "final_score": final_score,
            }
            results.append(flat_data)
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
            print("[OK] Success")

            client.files.delete(name=video_file.name)
            if temp_upload_path != source_path and temp_upload_path.exists():
                temp_upload_path.unlink()

        except Exception as e:
            print(f"[ERROR] {e}")
            if "temp_upload_path" in locals():
                try:
                    if temp_upload_path != Path(path) and temp_upload_path.exists():
                        temp_upload_path.unlink()
                except OSError:
                    pass

        time.sleep(5)

    print(f"\nDone. Results saved to: {output_csv}")


# ==========================================
# 執行區
# ==========================================
if __name__ == "__main__":
    API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    if not API_KEY:
        print("Please set GEMINI_API_KEY in environment variables.")
        raise SystemExit(1)

    TARGET_FOLDER = r"C:\AI影片辨識\test_videos"
    analyze_videos(TARGET_FOLDER, API_KEY)
