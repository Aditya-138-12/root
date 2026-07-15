## Project: HLS4ML Integration for fast ML Inference
I have solved this task and have added some tests too, the `/models` folder has 3 .onnx files, one is of the test, and other 2 are created to test the parser. Here I will write about how I have developed this parser and how to use it.

#### Parser Design
The parser is designed in such a way that, it takes a model stored in `.onnx` format and firstly runs some pre-check on this onnx model. The pre-checks are very important for the successfull
conversion of the model from onnx format to hls4ml in memory model. The reason is hls4ml internally uses a strict first approach while parsing the onnx model for conversion which basically means that for a particular layer in the onnx
format all of the attributes should be present for a successfull conversion. For example in a Conv2D layer, if dilations are set to 1x1, then onnx would omit this attribute while saving the model for saving some memory
and ONNX runtime is designed in such a way that it has no problem with the absent dilations attribute. But when converting this onnx model to hls4ml format, it will give you a error that
dilations is not present. So that is why we have to run some pre-checks for the supported layers so that we can add the ommited attributes with there defualt values.\
\
Also hls4ml internally uses the `channels-last` format for computations, therefore if a model has convolutional layers,then it has to be converted into channels-last format before conversion.\
HLS4ML has namely 5 backends that are: `Vivado`, `Vitis`, `Quartus`, `Catapult`, `oneAPI`, and the parser developed in this exercise supports all of them for parsing supported layers.
##### Supported Layers for parsing
1. Conv1D
2. Conv2D
3. Transpose
4. Reshape
5. Dense
6. Activation
7. Concatenate

#### Using parser
This parser takes some command lines arguments to which are indeed important for its functioning. `--modelPath` argument takes the relative path of the model stored in onnx file format, `--backend` argument takes the user specified backend for the hls4ml, `-L` is a flag, if set it prints just the layer wise configuration of the parsed hls4ml in memory model, `-IO` is another flag which if set, just prints the input and output layer configuration of the hls4ml in memory model.\
\
Usage: `python3 hls4ml_parser.py --backend Vivado --modelPath models/model.onnx -L -IO`

#### Tests
For running the tests, see below commands:\
`python3 hls4ml_parser.py --backend Quartus --modelPath models/model.onnx -L`\
`python3 hls4ml_parser.py --backend Vitis --modelPath models/model_comp.onnx -L`\
`python3 hls4ml_parser.py --backend Vivado --modelPath models/ConvWithAsymmetricPadding.onnx -L`\

#### References
1. https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/model/layers.py - tells which all layers are supported by hls4ml and what all functions each of the class supports..
2. https://github.com/fastmachinelearning/hls4ml/tree/main/hls4ml/backends - for working with the backend.

#### Update Regarding GSoC 2026
I did not get selected as a contributor for this project.
