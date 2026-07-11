from schemas.result import VideoResult

def render_json(result: VideoResult) -> None:
    """Dumps the VideoResult directly to standard output as formatted JSON."""
    print(result.model_dump_json(indent=2))
