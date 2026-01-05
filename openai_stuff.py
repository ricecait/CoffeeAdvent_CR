import openai
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai.api_key)

class CoffeeTasting(BaseModel):
    notes: List[str]

def extract_tasting_notes(message):

    if message == "":
        return CoffeeTasting(notes=[''])

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            #{"role": "system", "content": "Extract the main tasting notes or flavor descriptors from the following message."},
            {"role": "system",
             "content": "Extract the main scents or tasting notes or flavor descriptors from the following message. \
                         Strictly avoid adding smells or flavors that are not clearly present in the original. \
                         For example, for 'Got some white floral, jasmine early on and then some serious lemon acidity \
                         as it cooled, maybe a little white peach' you should return 'floral, jasmine, lemon acidity, \
                         white peach'. If the message is completely abstract like 'tastes like mowing the lawn', \
                         you should return something like 'grassy, fresh, loud'. Use your artistic license here but do \
                         not produce any notes or descriptors that can't be reasonably extracted from the message."},
            #{"role": "system", "content": "The following message describes a coffee tasting. Extract its main tasting notes and flavor descriptors."},
            {"role": "user", "content": message}
        ],
        response_format=CoffeeTasting,
        #max_tokens=1500,
        #temperature=0,
        #n=1,
    )
    # Accessing response as a Pydantic model
    tasting = response.choices[0].message.parsed

    return tasting

def get_embeddings(notes): # notes is list
    embeddings = []
    for note in notes:
        response = client.embeddings.create(
            model='text-embedding-3-large', #'text-embedding-ada-002',
            input=note.lower(),
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

if __name__ == '__main__':
    shane_str = """Wee little beans, prob Ethiopian. Light bodied. Got some white floral, jasmine early on and then some serious lemon acidity as it cooled, maybe a little white peach. Black tea, chamomile, lemon. I feel like I could use a cup of coffee to wash this down.

But Ben, in an attempt to try to pad my scores (and to make your work easier), I'll cut the fluff and just go with: black tea, chamomile, lemon.'"""

    mom_str = """On her way to Orange Beach to go shelling"""

    ben_str = """beans smell like mint chocolate, grounds smell like a deep, delicious umami chipotle bbq, coffee smelled like hot yoohoo. Taste: bright acidity. Main note by far is tomatillo. Maybe a little sweet cherry, but mostly just tomatillo. Also, I burnt my tongue last night on lava hot bone broth, so not sure I can trust anything about my tasting today."""

    notes = extract_tasting_notes(ben_str)
    print(notes)

    '''test_hoffman = open('./data/hoffman_tasting_notes.txt').readlines()

    for i in range(10):
        notes = extract_tasting_notes(test_hoffman[i])
        print(test_hoffman[i])
        print(notes)
        print()
        input()'''


