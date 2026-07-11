from schemas.result import VideoResult

def render(result: VideoResult) -> None:
    """Pretty-prints the completed VideoResult to the terminal using ANSI colors."""
    print("\n" + "=" * 70)
    print(f"\033[1;35mVideo Intelligence Report: {result.id}\033[0m")
    print(f"Source: \033[34m{result.source}\033[0m")
    print(f"Duration: \033[36m{result.duration:.2f}s\033[0m   Status: ", end="")
    
    if result.status == "ok":
        print("\033[1;32mOK (Success)\033[0m")
    elif result.status == "partial":
        print("\033[1;33mPARTIAL (Degraded execution)\033[0m")
    else:
        print("\033[1;31mFAILED\033[0m")
    print("=" * 70)

    # 1. Summary
    if result.summary:
        print(f"\n\033[1;33m[SUMMARY]\033[0m")
        print(result.summary)
        
    # 2. Detailed Summary
    if result.detailed_summary:
        print(f"\n\033[1;33m[DETAILED SUMMARY]\033[0m")
        print(result.detailed_summary)

    # 3. Tags
    if result.tags:
        print(f"\n\033[1;36m[TAGS]\033[0m")
        print(", ".join(f"#{t}" for t in result.tags))

    # 4. Timeline
    if result.timeline:
        print(f"\n\033[1;32m[CHRONOLOGICAL TIMELINE]\033[0m")
        for event in result.timeline:
            color = "\033[36m"  # Cyan for visual
            if event.source == "audio":
                color = "\033[32m"  # Green for speech
            elif event.source == "ocr":
                color = "\033[33m"  # Yellow for OCR
            print(f"  \033[1;30m[{event.time_display}]\033[0m {color}({event.source:<6})\033[0m {event.event}")

    # 5. Audio properties
    if result.audio:
        print(f"\n\033[1;34m[AUDIO ANALYSIS]\033[0m")
        print(f"  Speech Detected: {result.audio.speech}")
        music_info = f"Yes" + (f" ({result.audio.music_mood} mood)" if result.audio.music_mood else "") if result.audio.music else "No"
        print(f"  Music Present:   {music_info}")
        if result.audio.ambient:
            print(f"  Ambient Sounds:  {', '.join(result.audio.ambient)}")

    # 6. Scene segments info
    if result.scenes:
         print(f"\n\033[1;35m[SCENE BOUNDARIES]\033[0m")
         print(f"  Total Scene Segments Detected: {len(result.scenes)}")

    # 7. Degraded Stages
    if result.stage_errors:
        print(f"\n\033[1;31m[DEGRADED / FAILED PIPELINE STAGES]\033[0m")
        for stage, error in result.stage_errors.items():
            print(f"  - \033[1;31m{stage:<15}\033[0m: {error}")

    # 8. Performance timings
    if result.timings_ms:
        print(f"\n\033[1;30m[STAGE LATENCY METRICS]\033[0m")
        for stage, duration_ms in result.timings_ms.items():
            print(f"  - {stage:<15}: {duration_ms} ms")
            
    print("=" * 70 + "\n")
