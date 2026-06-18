import json

import uuid

from dataclasses import asdict

def save_architecture(
    architecture,
    directory
):

    architecture_id = str(
        uuid.uuid4()
    )

    filepath = (
        directory /
        f"{architecture_id}.json"
    )

    with open(filepath, "w") as f:

        json.dump(
            asdict(architecture),
            f,
            indent=4
        )

    return architecture_id