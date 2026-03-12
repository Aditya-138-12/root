## Project: HLS4ML Integration for fast ML Inference
I have solved this task and have added some tests in the `/tests` folder. Here I will write about how I have developed this parser and how to use it.
#### Using parser
This parser takes some command lines arguments to which are indeed important for its functioning. `--modelPath` argument takes the relative path of the model stored in onnx file format, `--backend` argument takes the user specified backend for the hls4ml, `-L` is a flag, if set it prints just the layer wise configuration of the parsed hls4ml in memory model, `-IO` is another flag which if set, just prints the input and output layer configuration of the hls4ml in memory model.\\
Usage: `python3 hls4ml_parser.py --backend Vivado --modelPath models/model.onnx -L -IO`

#### GenAI Decleration
The code in the file `hls4ml_parser.py` is not generated using any of the generative AI technologies like chatgpt/gemini/co-pilot, the algorithm design, complex problem solving, coding semantics, naming convention, implementation choices, etc are of the author i.e. Aditya.
