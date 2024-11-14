import argparse
import json
import multiprocessing as mp
import os

import ollama
import tqdm


class TextProcessor:
    def __init__(self, text_dir, json_save_dir=None) -> None:

        self.text_dir = text_dir
        self.text_filenames = [
            f for f in os.listdir(self.text_dir) if f.endswith(".txt")
        ]

        self.json_save_dir = json_save_dir
        if not os.path.exists(self.json_save_dir):
            os.makedirs(self.json_save_dir)

        self.json_template = json.dumps(
            {
                "companyName": "",
                "companyIdentifier": "",
                "documentPurpose": "",
            }
        )

        self.extract_prompt = """Given the following text extracted from a business document:
                    {text}
                    What is the Company Name, the Company Identifier, and the purpose of the document.
                    Give all your answers in English.
                    Use the following JSON template:
                    {json_template}"""

    def _load_text_from_path(self, text_path):
        with open(text_path, "r") as textfile:
            text = textfile.read()
        return text

    def _process_text(self, text_filename):
        text = self._load_text_from_path(os.path.join(self.text_dir, text_filename))

        prompt = self.extract_prompt.format(text=text, json_template=self.json_template)

        response = ollama.generate(
            model="llama3.2",
            prompt=prompt,
            format="json",
            options={"temperature": 0.0, "seed": 123},
        )["response"]

        if self.json_save_dir:
            json_save_path = os.path.join(
                self.json_save_dir, text_filename.replace(".txt", ".json")
            )
            with open(json_save_path, "w") as outfile:
                outfile.write(response)

        return response

    def run(self):
        data = []
        with mp.Pool() as pool:
            for result in tqdm.tqdm(
                pool.imap(self._process_text, self.text_filenames),
                total=len(self.text_filenames),
            ):
                data.append(result)

        return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", type=str, required=True)
    parser.add_argument("--json_save_dir", type=str, required=False)
    args = parser.parse_args()

    processor = TextProcessor(text_dir=args.text_dir, json_save_dir=args.json_save_dir)
    data = processor.run()
