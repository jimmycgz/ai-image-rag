import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
import os
import json
from pydantic import BaseModel
from typing import Tuple

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Molmo model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

class GeneralRetrievalQuery(BaseModel):
    broad_topical_query: str
    broad_topical_explanation: str
    specific_detail_query: str
    specific_detail_explanation: str
    visual_element_query: str
    visual_element_explanation: str

def get_retrieval_prompt(prompt_name: str) -> Tuple[str, GeneralRetrievalQuery]:
    if prompt_name != "general":
        raise ValueError("Only 'general' prompt is available in this version")

    prompt = """You are an AI assistant specialized in document retrieval tasks. Given an image of a document page, your task is to generate retrieval queries that someone might use to find this document in a large corpus.

Please generate 3 different types of retrieval queries:

1. A broad topical query: This should cover the main subject of the document.
2. A specific detail query: This should focus on a particular fact, figure, or point made in the document.
3. A visual element query: This should reference a chart, graph, image, or other visual component in the document, if present. Don't just reference the name of the visual element but generate a query which this illustration may help answer or be related to.

Important guidelines:
- Ensure the queries are relevant for retrieval tasks, not just describing the page content.
- Frame the queries as if someone is searching for this document, not asking questions about its content.
- Make the queries diverse and representative of different search strategies.

For each query, also provide a brief explanation of why this query would be effective in retrieving this document.

Format your response as a JSON object with the following structure:

{
  "broad_topical_query": "Your query here",
  "broad_topical_explanation": "Brief explanation",
  "specific_detail_query": "Your query here",
  "specific_detail_explanation": "Brief explanation",
  "visual_element_query": "Your query here",
  "visual_element_explanation": "Brief explanation"
}

If there are no relevant visual elements, replace the third query with another specific detail query.

Here is the document image to analyze:

Generate the queries based on this image and provide the response in the specified JSON format.
Only return JSON. Don't return any extra explanation text. """

    return prompt, GeneralRetrievalQuery

prompt, pydantic_model = get_retrieval_prompt("general")

def _prep_data_for_input(image):
    return processor.process(
        images=[image],
        text=prompt,
        return_tensors="pt"
    )

def generate_response(image):
    inputs = _prep_data_for_input(image)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_text = processor.tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        return str(json.loads(generated_text))
    except json.JSONDecodeError:
        gr.Warning("Failed to parse JSON from output")
        return generated_text

title = "ColPali fine-tuning Query Generator"
description = """[ColPali](https://huggingface.co/papers/2407.01449) is a very exciting new approach to multimodal document retrieval which aims to replace existing document retrievers which often rely on an OCR step with an end-to-end multimodal approach. 

To train or fine-tune a ColPali model, we need a dataset of image-text pairs which represent the document images and the relevant text queries which those documents should match. 
To make the ColPali models work even better we might want a dataset of query/image document pairs related to our domain or task. 

One way in which we might go about generating such a dataset is to use a VLM to generate synthetic queries for us. 
This space uses the [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) model to generate queries for a document, based on an input document image. 

**Note** there is a lot of scope for improving to prompts and the quality of the generated queries! If you have any suggestions for improvements please [open a Discussion](https://huggingface.co/spaces/davanstrien/ColPali-Query-Generator/discussions/new)!

This [blog post](https://danielvanstrien.xyz/posts/post-with-code/colpali/2024-09-23-generate_colpali_dataset.html) gives an overview of how you can use this kind of approach to generate a full dataset for fine-tuning ColPali models. 

If you want to convert a PDF(s) to a dataset of page images you can try out the [ PDFs to Page Images Converter](https://huggingface.co/spaces/Dataset-Creation-Tools/pdf-to-page-images-dataset) Space.
"""

examples = [
    "examples/Approche_no_13_1977.pdf_page_22.jpg",
    "examples/SRCCL_Technical-Summary.pdf_page_7.jpg",
]

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Image(type="pil"),
    outputs=gr.Text(),
    title=title,
    description=description,
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()