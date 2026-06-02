# Meeting Health Dashboard

Local runnable files for VS Code.

## Files

- `notebooks/Meeting_Health_Dashboard_fixed.ipynb`: runs the meeting intelligence pipeline.
- `dashboard/app.py`: runs the Streamlit dashboard.
- `dashboard/data/`: generated CSVs go here after the notebook runs.
- `requirements.txt`: Python dependencies.

## Setup

Open this folder in VS Code:

```bash
cd "/Users/saiteja/Documents/New project/meeting_health_dashboard_local"
python3 -m pip install -r requirements.txt
```

Set your API key in the terminal before running the notebook or dashboard:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

You can also put the key in a local `.env` file at the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

## Run The Notebook

```bash
cd "/Users/saiteja/Documents/New project/meeting_health_dashboard_local/notebooks"
jupyter notebook
```

Open `Meeting_Health_Dashboard_fixed.ipynb` and run all cells.

The notebook also supports a faster demo run by default. For full-length audio processing:

```bash
export FULL_MEETING_RUN=1
```

## Run The Dashboard

In a second terminal:

```bash
cd "/Users/saiteja/Documents/New project/meeting_health_dashboard_local/dashboard"
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

Open the URL Streamlit prints, usually:

```text
http://localhost:8501
```

## Notes

- The dashboard loads generated files automatically from `dashboard/data`.
- Manual uploads are optional overrides only.
- Keep your API key in the terminal environment. Do not paste it into code.

## AI Meeting Assistant - Feature 2 Audio Setup

The AI Meeting Assistant is designed to attend or process meetings on behalf of the user. It can listen, take notes, track decisions and action items, privately advise the user, and in `FULL_PROXY` mode speak in first person as the user within the authority and style provided in the briefing.

Recommended proxy briefing fields:

- your name and role
- a disclosed assistant display name, such as `Sai - AI Assistant`
- how you naturally speak
- what the assistant may decide or commit to
- what it must defer back to you
- talking points it should raise
- topics it must avoid

Feature 2 adds live audio capture, live Whisper transcription, and speaker tracking modules under `assistant/`.

Install assistant dependencies:

```bash
cd "/Users/saiteja/Desktop/meeting-health-dashboard"
python3 -m pip install -r requirements_assistant.txt
```

The default install uses `faster-whisper`. `openai-whisper` is kept as an optional fallback because its source build can fail on some Python/setuptools combinations:

```bash
python3 -m pip install "setuptools<81" wheel
python3 -m pip install --no-build-isolation -r requirements_assistant_optional.txt
```

The assistant captures audio with `sounddevice` by default. `pyaudio` is optional because it requires PortAudio headers on macOS:

```bash
brew install portaudio
python3 -m pip install -r requirements_assistant_audio_optional.txt
```

Install `ffmpeg` before processing audio files:

```bash
brew install ffmpeg
```

For system audio capture on macOS, install BlackHole and create a Multi-Output Device in Audio MIDI Setup that includes both your speakers and BlackHole:

```bash
brew install blackhole-2ch
```

Then set this in `.env`:

```text
VIRTUAL_AUDIO_DEVICE_NAME=BlackHole 2ch
```

For Windows, install VB-Cable and set:

```text
VIRTUAL_AUDIO_DEVICE_NAME=CABLE Output (VB-Audio Virtual Cable)
```

For Linux with PulseAudio, create a null sink:

```bash
pactl load-module module-null-sink sink_name=meeting_monitor sink_properties=device.description=meeting_monitor
pactl set-default-sink meeting_monitor
```

Then set:

```text
VIRTUAL_AUDIO_DEVICE_NAME=meeting_monitor.monitor
```

Optional diarization uses pyannote. Add `HF_TOKEN` to `.env` after creating a Hugging Face token and accepting the pyannote model terms. If `HF_TOKEN` is missing, speaker tracking falls back to a local single-speaker energy detector so the meeting loop can continue.

Feature 3 adds the meeting agent brain, private advisor tips, runtime memory, and response playback. Add these optional values to `.env` if you want spoken responses:

```text
OPENAI_AGENT_MODEL=gpt-4o-mini
OPENAI_MEMORY_MODEL=gpt-4o-mini
OPENAI_FLAG_MODEL=gpt-4o-mini
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
SILENCE_RMS_THRESHOLD=350
MEMORY_UPDATE_INTERVAL_SECONDS=300
```

If ElevenLabs or local TTS is unavailable, the responder prints the text to the terminal and logs the response instead of crashing.

Feature 4 adds real-time private flags and a do-not-miss checklist. The agent flags likely decisions, action items, open questions, tension spikes, topic shifts, and talking-point opportunities. The checklist is built from the meeting objective, talking points, and any custom instruction such as "make sure we cover pricing." At the end of a meeting, any pending item can be marked missed with a suggested follow-up.

Feature 5 adds recording export. The recorder saves each meeting under `RECORDINGS_DIR/{session_id}/` with:

- `audio_raw.wav` for the existing analysis pipeline
- `audio.mp3` for compact playback when `ffmpeg` is installed
- `transcript.json`, `transcript.txt`, and `transcript.srt`
- `session_metadata.json`

Session IDs use `YYYYMMDD_HHMM_meeting_title_slug`, for example `20260529_1430_q3_planning_review`.

Feature 6 adds assistant-session pipeline loading. After a session has `transcript.json`, run:

```bash
cd "/Users/saiteja/Desktop/meeting-health-dashboard"
python3 -c "from post_meeting.pipeline_runner import run_existing_pipeline; print(run_existing_pipeline('YOUR_SESSION_ID'))"
```

This writes dashboard-compatible CSVs into `recordings/YOUR_SESSION_ID/pipeline/`. In the Streamlit sidebar, use **Load from Assistant Session** to select the session. The normal dashboard charts will load those CSVs, and the **Assistant Session** tab will show metadata, checklist status, agent activity, key moments, audio, and transcript.

The **Assistant Control** tab gives you a browser-based setup surface for testing the assistant. You can upload a TXT, Markdown, PDF, or DOCX resume, generate a meeting briefing for an advisor or proxy interview run, save that briefing, upload recorded audio for file-mode testing, and browse saved assistant session files. Proxy mode uses a transparent display name such as `Sai - AI Assistant`; the assistant can answer from your authorized resume/project context without pretending to be the human user. PDF and DOCX extraction use `pypdf` and `python-docx` from `requirements_assistant.txt`.

The Ask tab also reports RAG answer-quality metrics after each answer:

- **Hallucination**: estimated percentage of answer content not supported by the retrieved context.
- **Faithfulness**: estimated percentage of answer claims supported by the retrieved context.
- **Meeting Relevance**: estimated percentage of answer content relevant to the original meeting transcript.

When an API key is available, a strict LLM judge scores these three metrics. If the judge is unavailable, the dashboard falls back to lexical overlap estimates. The details panel also shows query coverage, retrieved-context relevance, citation coverage, and per-document matched terms.

Feature 7 adds the session manager and CLI:

```bash
cd "/Users/saiteja/Desktop/meeting-health-dashboard"
python3 -m assistant.cli
```

During a live meeting, type commands with a `!` prefix:

```text
!status
!raise 1
!tip
!speak I want to clarify the owner before we move on.
!mute
!unmute
!flag Important customer escalation
!checklist
!whoisspeaking
!end
```

For a pre-recorded file, choose `audio file` in the CLI. The session manager converts supported audio to 16kHz mono WAV, transcribes chunks, updates memory/checklist state, saves recordings/transcripts/metadata, and runs the dashboard pipeline.

Virtual meeting integrations are available for Zoom, Microsoft Teams, and Google Meet:

- Zoom opens meeting links through the Zoom URL scheme. SSO or 2FA may require manual sign-in.
- Teams opens meeting links through the OS URL handler. Work-account SSO or MFA may require manual sign-in.
- Google Meet uses Selenium with undetected-chromedriver when installed and configured. Bot detection may still block login.

All integrations rely on virtual audio routing. Test routing with:

```bash
python3 -m integrations.audio_router test
```

Set optional credentials in `.env` only if you need automated sign-in:

```text
ZOOM_EMAIL=
ZOOM_PASSWORD=
GOOGLE_EMAIL=
GOOGLE_PASSWORD=
TEAMS_EMAIL=
TEAMS_PASSWORD=
```

Feature 8 adds post-meeting reports and notifications. Reports are always saved as:

- `recordings/YOUR_SESSION_ID/report.md`
- `recordings/YOUR_SESSION_ID/report.json`

Optional notification settings:

```text
OPENAI_REPORT_MODEL=gpt-4o-mini
SLACK_WEBHOOK_URL=
SENDGRID_API_KEY=
REPORT_EMAIL=
REPORT_EMAIL_FROM=
```

Build a report manually:

```bash
python3 -c "from assistant.briefing import Briefing, AgentMode; from assistant.memory import RuntimeMemory; from post_meeting.pipeline_runner import run_existing_pipeline; from post_meeting.report_builder import build_report; sid='YOUR_SESSION_ID'; briefing=Briefing('Meeting','Review outcomes',[],'observer',AgentMode.SILENT_OBSERVER,[],[],'','concise',[],0,False); outputs=run_existing_pipeline(sid); report=build_report(sid, RuntimeMemory(), briefing, outputs); print(report.markdown_path)"
```

Quick smoke test:

```bash
cd "/Users/saiteja/Desktop/meeting-health-dashboard"
python3 -c "from assistant.audio_capture import AudioCapture; from assistant.transcriber import TranscriptSegment; from assistant.speaker_tracker import SpeakerTracker; print(AudioCapture().chunk_seconds, TranscriptSegment(0, 1, 'ok').text, SpeakerTracker().current_active_speaker())"
```

Agent smoke test:

```bash
cd "/Users/saiteja/Desktop/meeting-health-dashboard"
python3 -c "from assistant.briefing import Briefing, AgentMode; from assistant.memory import learn_from_briefing; from assistant.agent import MeetingAgent; from assistant.responder import Responder; from assistant.transcriber import TranscriptSegment; from assistant.speaker_tracker import SpeakerTracker; b=Briefing('Roadmap','Align launch',['Sam'],'lead',AgentMode.PARTICIPATOR,['budget risk'],['layoffs'],'Budget risk matters.','concise',[],3,True); mem=learn_from_briefing(b); tr=type('T',(),{'get_recent_context':lambda self, seconds=120:'Sam: Can you talk about the budget risk?'})(); agent=MeetingAgent(b,mem,tr,SpeakerTracker(),Responder()); print(agent.should_speak('Sam: Can you talk about the budget risk?'))"
```

Checklist smoke test:

```bash
cd "/Users/saiteja/Desktop/meeting-health-dashboard"
python3 -c "from assistant.briefing import Briefing, AgentMode; from assistant.session import DoNotMissChecklist; from assistant.transcriber import TranscriptSegment; b=Briefing('Roadmap','Align launch',['Sam'],'lead',AgentMode.SILENT_OBSERVER,['budget risk'],[],'','concise',['make sure ownership is confirmed'],0,False); c=DoNotMissChecklist.from_briefing(b); c.update_from_segment(TranscriptSegment(1, 2, 'We agreed the budget risk owner is Sam.')); c.print_live_panel()"
```

Recorder smoke test:

```bash
cd "/Users/saiteja/Desktop/meeting-health-dashboard"
python3 -c "from pathlib import Path; from assistant.recorder import Recorder, generate_session_id; from assistant.transcriber import TranscriptSegment; sid=generate_session_id('Recorder Smoke'); r=Recorder(Path('/tmp/meeting-recorder-smoke')); r.start_recording(sid); r.write_chunk(b'\\0\\0' * 16000); print(r.stop_recording()); print(r.save_transcript([TranscriptSegment(0, 1, 'hello', 'Speaker_A')], sid))"
```

## AI Meeting Assistant Limitations

- Google Meet may block automated browser login or require manual verification.
- Zoom SDK-level joining requires developer credentials; this implementation uses the desktop URL scheme and manual fallback.
- Teams work accounts commonly require SSO or MFA.
- System audio capture requires BlackHole, VB-Cable, or PulseAudio null sink setup.
- LLM calls can cost money. At the default settings, occasional briefing/report/summary calls are usually low cost, while frequent live comprehension on long meetings costs more. Use offline fallbacks by leaving API keys unset.

## AI Meeting Assistant Delivery Checklist

- [x] `assistant/` core files implemented
- [x] `integrations/` Zoom, Teams, Google Meet, and audio router files implemented
- [x] `post_meeting/` pipeline runner, report builder, and notifier implemented
- [x] `dashboard_ext/assistant_tab.py` implemented
- [x] `.env.example` documents all assistant keys
- [x] `requirements_assistant.txt` lists assistant dependencies
- [x] README assistant setup, examples, and limitations documented
- [x] Dashboard can load assistant session pipeline outputs
- [x] LLM calls have rule-based fallbacks
- [x] Supported audio formats normalize through ffmpeg for file mode
- [x] JSON/TXT/SRT/report/metadata saves use atomic writes
- [x] Session tasks handle cancellation on stop
- [x] Virtual audio errors include setup guidance
- [x] KeyboardInterrupt in CLI triggers cleanup path
