
# import sys
# import os
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# sys.path.insert(0, PROJECT_ROOT)

import asyncio
from serve import serve_validate_m_test

# prompt = "A simple, minimalist padlock with a circular dial mechanism, rendered in a clean, monochrome style."
prompt = "A set of wooden bowling pins and a red bowling ball on a wooden surface."

async def test_scoring():
    output_folder = "/workspace/Bittensor/neural-subnet/generate/outputs/text_to_3d/"
    await serve_validate_m_test(f"{output_folder}mesh.glb", f"{output_folder}mesh.png", prompt=prompt)


if __name__ == "__main__":
    asyncio.run(test_scoring())