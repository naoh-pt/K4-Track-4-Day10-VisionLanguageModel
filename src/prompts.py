PROMPT = (
    'Locate all potholes, cracks, and manholes in this road image. '
    'Return only a JSON list, with no explanation or markdown. '
    'Each object must have "label" ("pothole", "crack", or "manhole") and '
    '"bbox_2d": [x1,y1,x2,y2]. Use integer coordinates normalized to 0–1000 '
    'relative to image width and height. Sort by top edge, then left edge. '
    'If there are no objects, return [].'
)


def user_message(image):
    return {"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}
