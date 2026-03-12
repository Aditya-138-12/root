## Project: HLS4ML Integration for fast ML Inference
I have solved this task and have added some tests in the `/tests` folder. Here I will write about how I have developed this parser and how to use it.

#### Parser Design
The parser is designed in such a way that, it takes a model stored in `.onnx` format and firstly runs some pre-check on this onnx model. The pre-checks are very important for the successfull\
conversion of the model from onnx format to hls4ml in memory model. The reason is hls4ml internally uses a strict first approach which basically means that for a particular layer in the onnx\
format all of the attributes should be present. For example in a Conv2D layer, if dilations are set to 1x1, then onnx would omit this attribute while saving the model for saving some memory\
and ONNX runtime is designed in such a way that it has no problem with the absent dilations attribute. But when converting this onnx model to hls4ml format, it will give you a error that\
dilations is not present. So that is why we have to run some pre-checks for the supported layers so that we can add the ommited attributes with there defualt values.

#### Using parser
This parser takes some command lines arguments to which are indeed important for its functioning. `--modelPath` argument takes the relative path of the model stored in onnx file format, `--backend` argument takes the user specified backend for the hls4ml, `-L` is a flag, if set it prints just the layer wise configuration of the parsed hls4ml in memory model, `-IO` is another flag which if set, just prints the input and output layer configuration of the hls4ml in memory model.\
\
Usage: `python3 hls4ml_parser.py --backend Vivado --modelPath models/model.onnx -L -IO`

#### GenAI Decleration
The code in the file `hls4ml_parser.py` is not generated using any of the generative AI technologies like chatgpt/gemini/co-pilot, the algorithm design, complex problem solving, coding semantics, naming convention, implementation choices, etc are of the author i.e. Aditya.
