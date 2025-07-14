import bittensor as bt
import time
import asyncio
import aiohttp


prompt = "A set of wooden bowling pins and a red bowling ball on a wooden surface."
# prompt = "A cat"

async def test_generate_3D_request():
    """
    Test the 3D mesh generation request by sending a sample prompt.
    """
    gen_url = "http://127.0.0.1:8093/generate_from_text/"  # Adjust the URL as needed
    client_timeout = aiohttp.ClientTimeout(total=60)  # Safer default timeout

    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        try:
            bt.logging.debug(f"Sending request to {gen_url}")
            async with session.post(gen_url, data={"prompt": prompt}) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Success:", result)
                    return result
                else:
                    bt.logging.error(f"❌ Generation failed. Status code: {response.status}")
        except Exception as e:
            bt.logging.error(f"🚨 Exception during request: {e}")

if __name__ == "__main__":
    asyncio.run(test_generate_3D_request())

