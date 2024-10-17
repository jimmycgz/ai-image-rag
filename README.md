# ColPali fine-tuning Query Generator
Original repo https://huggingface.co/spaces/davanstrien/ColPali-Query-Generator

## Using the Application:

   a. The interface will allow you to upload an image of a document.

   b. The model will generate queries based on the document content.

   c. Results will be displayed in the interface.

## Performance Considerations:

   * This setup runs the model on CPU, which will be significantly slower than GPU acceleration.

   * For better performance, consider using a smaller model or exploring cloud-based GPU solutions.

Remember, running large language models on a VM without GPU acceleration will be slow. This setup is more suitable for testing and development rather than production use.

## Deployment

* `Deploy-Mac.md` for MacOS, tested on M3 Max
* `Deploy-win11.md` tested win11 on UTM in M3, can only use cpu and extremely slow

## Troubleshooting:

   * If you encounter "out of memory" errors, try closing other applications or restarting your VM.
   * For disk space issues, consider expanding your VM's storage or removing unnecessary files.
   * If the model loading is extremely slow, be patient as it's running on CPU.

