# AI Video Summarizer & Quiz Generator

An AI-powered web application that transforms educational videos into
searchable transcripts, structured summaries, key topics, and
interactive multiple-choice quizzes.

## Overview

The application helps users review video-based content efficiently. It
processes an uploaded video, extracts its audio, converts speech into
text, and uses AI to generate learning materials from the transcript.
Quiz results and processed-video history are stored locally.

## Features

-   **Video upload:** Supports MP4, MOV, WEBM, AVI, and MKV formats,
    with drag-and-drop upload and client-side validation.
-   **Audio extraction:** Uses FFmpeg to extract audio from uploaded
    videos.
-   **Speech transcription:** Uses Whisper for speech-to-text
    transcription.
-   **AI summaries:** Generates structured summaries and key points from
    the transcript using Google Gemini.
-   **Topic extraction:** Identifies important topics and presents them
    as descriptive concept cards.
-   **MCQ generation:** Creates multiple-choice questions based on the
    transcript, with four answer options.
-   **Configurable quizzes:** Offers Easy, Medium, and Hard difficulty
    levels and selectable question counts.
-   **Quiz assessment:** Provides progress tracking, scores, answer
    review, explanations, and AI-generated study recommendations.
-   **Learning history:** Uses SQLite to retain processed-video details
    and quiz scores.
-   **Demo mode:** Includes sample educational content so the
    application can be demonstrated without a Gemini API key.
-   **Responsive interface:** Supports desktop, tablet, and mobile
    layouts, with light and dark themes.

## Technology Stack

  Layer                  Technologies
  ---------------------- ----------------------------------------
  Frontend               HTML5, CSS3, JavaScript (Vanilla ES6+)
  Backend                Python, Flask
  AI and summarization   Google Gemini API
  Speech-to-text         Whisper
  Media processing       FFmpeg
  Database               SQLite
  Templates              Jinja2

## Application Workflow

1.  Upload an educational video.
2.  The Flask application validates and stores the uploaded file.
3.  FFmpeg extracts the audio track.
4.  Whisper converts the audio into a transcript.
5.  The transcript is processed by Google Gemini to generate a summary,
    key points, topics, and quiz questions.
6.  The application displays the generated learning materials and
    interactive quiz.
7.  Video details and quiz results are saved in the SQLite database.

## Project Structure

``` text
project/
├── app.py
├── demo_data.py
├── nlp_engine.py
├── requirements.txt
├── .env.example
├── .gitignore
├── database/
├── uploads/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── processing.html
│   ├── result.html
│   ├── quiz_setup.html
│   ├── quiz.html
│   ├── score.html
│   └── history.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

The exact files and folders may vary slightly depending on the current
project version. Generated files, local databases, uploaded videos,
virtual environments, and secrets should not be committed unless they
are intentionally part of the source code.

## Requirements

-   Python 3.9 or later
-   pip
-   FFmpeg installed and available on the system PATH for video audio
    extraction
-   A Google Gemini API key for AI-powered generation (optional when
    using Demo Mode)

## Installation

1.  Clone the repository and move into the project directory:

    ``` bash
    git clone <repository-url>
    cd <project-folder>
    ```

2.  Create a virtual environment:

    ``` bash
    python -m venv venv
    ```

3.  Activate the environment.

    **Windows:**

    ``` bash
    venv\Scripts\activate
    ```

    **Linux/macOS:**

    ``` bash
    source venv/bin/activate
    ```

4.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

5.  Configure environment variables. Copy `.env.example` to `.env` and
    add a Gemini API key if AI features require it.

    Example `.env.example`:

    ``` env
    GEMINI_API_KEY=
    SECRET_KEY=replace-with-a-local-secret
    MAX_UPLOAD_SIZE_MB=2048
    ```

    Keep the real `.env` file private. Never commit API keys, passwords,
    or other secrets to GitHub.

## Run the Application

Start the Flask application:

``` bash
python app.py
```

Open the local address shown by Flask in a browser. The default
development address is commonly:

``` text
http://127.0.0.1:5000
```

## Demo Mode

The application includes a demo workflow with preconfigured educational
content. This can be used to explore the interface when a Gemini API key
is not configured. Features that require external AI services may be
limited in Demo Mode.

## Security and Repository Notes

-   Do not upload `.env` or any file containing real API keys or
    credentials.
-   Do not commit the `venv/` directory.
-   Avoid committing large uploaded video files or generated runtime
    data.
-   Keep `.env.example` limited to placeholder values.
-   Review files before making the repository public.

## Future Enhancements

-   Synchronize summaries and quiz questions with video timestamps.
-   Add support for more languages and translation.
-   Export summaries as PDF and generate flashcards.
-   Add expanded progress analytics and learning dashboards.
