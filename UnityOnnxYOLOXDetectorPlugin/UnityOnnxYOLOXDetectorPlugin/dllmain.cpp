// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <onnxruntime_cxx_api.h>
#include "dml_provider_factory.h"

// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {
	// The current model input width
	int input_w;
	// The current model input height
	int input_h;
	// The total number pixels in the input image
	int n_pixels;
	// The number of color channels 
	int n_channels = 3;

	// ONNX Runtime API interface
	const OrtApi* ort{ nullptr };

	// List of available execution providers
	char** provider_names;
	int provider_count;

	// Holds the logging state for the ONNX Runtime objects
	OrtEnv* env;
	// Holds the options used when creating a new ONNX Runtime session
	OrtSessionOptions* session_options;
	// The ONNX Runtime session
	OrtSession* session;

	// The name of the model input
	char* input_name;
	// The name of the model output
	char* output_name;

	// A pointer to the raw input data
	float* input_data;
	// The memory size of the raw input data
	int input_size;


	/// <summary>
	/// Convert char data to wchar_t
	/// </summary>
	/// <param name="text"></param>
	/// <returns></returns>
	static wchar_t* charToWChar(const char* text)
	{
		const size_t size = strlen(text) + 1;
		wchar_t* wText = new wchar_t[size];
		size_t converted_chars;
		mbstowcs_s(&converted_chars, wText, size, text, _TRUNCATE);
		return wText;
	}


	/// <summary>
	/// Initialize the ONNX Runtime API interface and get the available execution providers
	/// </summary>
	/// <returns></returns>
	DLLExport void InitOrtAPI() {

		ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

		ort->GetAvailableProviders(&provider_names, &provider_count);
	}


	/// <summary>
	/// Get the number of available execution providers
	/// </summary>
	/// <returns>The number of available devices</returns>
	DLLExport int GetProviderCount()
	{
		// Return the number of available execution providers
		return provider_count;
	}


	/// <summary>
	/// Get the name of the execution provider at the specified index
	/// </summary>
	/// <param name="index"></param>
	/// <returns>The name of the execution provider at the specified index</returns>
	DLLExport char* GetProviderName(int index) {
		return provider_names[index];
	}

	/// <summary>
	/// Refresh memory when switching models or execution providers
	/// </summary>
	DLLExport void RefreshMemory() {
		if (input_data) free(input_data);
		if (session) ort->ReleaseSession(session);
		if (env) ort->ReleaseEnv(env);
	}

	/// <summary>
	/// Load a model from the specified file path
	/// </summary>
	/// <param name="model_path">The full model path to the ONNX model</param>
	/// <param name="execution_provider">The name for the desired execution_provider</param>
	/// <param name="image_dims">The source image dimensions</param>
	/// <returns>A status value indicating success or failure to load and reshape the model</returns>
	DLLExport int LoadModel(char* model_path, char* execution_provider, int image_dims[2])
	{
		int return_val = 0;

		// Initialize the ONNX runtime environment
		std::string instance_name = "yolox-inference";
		ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, instance_name.c_str(), &env);

		// Disable telemetry
		ort->DisableTelemetryEvents(env);

		// Add the selected execution provider
		ort->CreateSessionOptions(&session_options);
		std::string provider_name = execution_provider;

		// Add the specified execution provider
		if (provider_name.find("CPU") != std::string::npos) {
			return_val = 1;
		}
		else if (provider_name.find("Dml") != std::string::npos) {
			ort->DisableMemPattern(session_options);
			ort->SetSessionExecutionMode(session_options, ExecutionMode::ORT_SEQUENTIAL);
			OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);
		}
		else return_val = 1;

		// Create a new inference session
		ort->CreateSession(env, charToWChar(model_path), session_options, &session);
		ort->ReleaseSessionOptions(session_options);

		Ort::AllocatorWithDefaultOptions allocator;

		// Get input and output names
		ort->SessionGetInputName(session, 0, allocator, &input_name);
		ort->SessionGetOutputName(session, 0, allocator, &output_name);

		// The dimensions of the input image
		input_w = image_dims[0];
		input_h = image_dims[1];
		
		n_pixels = input_w * input_h;

		// Allocate memory for the raw input data
		input_size = n_pixels * n_channels * (int)sizeof(float);
		input_data = (float*)malloc((size_t)input_size * sizeof(float*));
		if (input_data != NULL) memset(input_data, 0, input_size);

		// Return a value of 0 if the model loads successfully
		return return_val;
	}


	/// <summary>
	/// Perform inference with the provided texture data
	/// </summary>
	/// <param name="image_data">The source image data from Unity</param>
	/// <returns>The final number of detected objects</returns>
	DLLExport void PerformInference(byte* image_data, float* output_array, int length)
	{
		// Iterate over each pixel in image
		for (int p = 0; p < n_pixels; p++)
		{
			for (int ch = 0; ch < n_channels; ch++) {
				// Scale and normalize each value
				input_data[ch * n_pixels + p] = (image_data[p * n_channels + ch] / 255.0f);
			}
		}

		// Initialize list of input and output names
		const char* input_names[] = { input_name };
		const char* output_names[] = { output_name };
		// Initialize the list of model input dimension
		int64_t input_shape[] = { 1, 3, input_h, input_w };
		int input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

		// Initialize an input tensor object with the input_data
		OrtMemoryInfo* memory_info;
		ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

		OrtValue* input_tensor = NULL;
		ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, input_size, input_shape,
			input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			&input_tensor);

		ort->ReleaseMemoryInfo(memory_info);


		OrtValue* output_tensor = NULL;
		// Perform inference
		ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
			&output_tensor);

		// Make sure the output tensor is not NULL to avoid potential crashes
		if (output_tensor == NULL) {
			ort->ReleaseValue(input_tensor);
			ort->ReleaseValue(output_tensor);
			return;
		}

		// Get the length of a single object proposal (i.e., number of object classes + 5)
		OrtTensorTypeAndShapeInfo* output_tensor_info;
		ort->GetTensorTypeAndShape(output_tensor, &output_tensor_info);
		size_t output_length[1] = {};
		ort->GetDimensionsCount(output_tensor_info, output_length);
		int64_t output_dims[3] = {};
		ort->GetDimensions(output_tensor_info, output_dims, *output_length);

		// Access model output
		float* out_data;
		ort->GetTensorMutableData(output_tensor, (void**)&out_data);

		// Copy model output to the output array
		std::memcpy(output_array, out_data, length * sizeof(float));

		// Free memory for input and output tensors
		ort->ReleaseValue(input_tensor);
		ort->ReleaseValue(output_tensor);
	}


	/// <summary>
	/// Free memory
	/// </summary>
	DLLExport void FreeResources()
	{
		free(input_data);
		ort->ReleaseSession(session);
		ort->ReleaseEnv(env);
	}
}