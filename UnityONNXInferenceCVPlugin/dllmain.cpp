#include "pch.h"
#include <onnxruntime_cxx_api.h>
#include "dml_provider_factory.h"
#include <string>
#include <vector>
#include <functional>

#define DLLExport __declspec (dllexport)

extern "C" {
	int input_w;                      // Width of the input image
	int input_h;                      // Height of the input image
	int n_pixels;                     // Total number of pixels in the input image (width x height)
	const int n_channels = 3;         // Number of color channels in the input image (3 for RGB)
	const OrtApi* ort = nullptr;      // Pointer to the ONNX Runtime C API, used for most ONNX operations
	std::vector<std::string> provider_names; // Names of providers available for ONNX runtime, e.g., DirectML
	OrtEnv* env = nullptr;            // ONNX Runtime environment, encapsulating global options and logging functionality
	OrtSessionOptions* session_options = nullptr; // Configurations and settings for the ONNX session
	OrtSession* session = nullptr;    // The ONNX Runtime session, representing the loaded model and its state
	std::string input_name;           // Name of the model's input node
	std::string output_name;          // Name of the model's output node
	std::vector<float> input_data;    // Buffer to hold preprocessed input data before feeding it to the model

	/// <summary>
	/// Convert a standard string to a wide string.
	/// </summary>
	/// <param name="str">A standard string to convert.</param>
	/// <returns>The wide string representation of the given string.</returns>
	std::wstring stringToWstring(const std::string& str) {
		std::wstring wstr(str.begin(), str.end());
		return wstr;
	}

	/// <summary>
	/// Initialize the ONNX Runtime API and retrieve the available providers.
	/// </summary>
	/// <returns></returns>
	DLLExport void InitOrtAPI() {
		// Get the ONNX Runtime API
		ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

		// Temporary pointers to fetch provider names
		char** raw_provider_names;
		int provider_count;

		// Get available providers
		ort->GetAvailableProviders(&raw_provider_names, &provider_count);

		// Populate the global provider names vector
		provider_names = std::vector<std::string>(raw_provider_names, raw_provider_names + provider_count);
	}

	/// <summary>
	/// Get the number of available execution providers.
	/// </summary>
	/// <returns>The number of execution providers.</returns>
	DLLExport int GetProviderCount() {
		return static_cast<int>(provider_names.size());
	}

	/// <summary>
	/// Retrieve the name of a specific execution provider.
	/// </summary>
	/// <param name="index">The index of the provider.</param>
	/// <returns>The name of the provider or nullptr if index is out of bounds.</returns>
	DLLExport const char* GetProviderName(int index) {
		if (index >= 0 && index < provider_names.size()) {
			return provider_names[index].c_str();
		}
		return nullptr;
	}

	/// <summary>
	/// Release allocated ONNX Runtime resources.
	/// </summary>
	/// <returns></returns>
	DLLExport void FreeResources() {
		if (session) ort->ReleaseSession(session);
		if (env) ort->ReleaseEnv(env);
	}
	
	/// <summary>
	/// Load an ONNX model and prepare it for inference.
	/// </summary>
	/// <param name="model_path">Path to the ONNX model file.</param>
	/// <param name="execution_provider">The execution provider to use (e.g., "CPU" or "Dml").</param>
	/// <param name="image_dims">Dimensions of the input image [width, height].</param>
	/// <returns>A message indicating the success or failure of the loading process.</returns>
	DLLExport const char* LoadModel(const char* model_path, const char* execution_provider, int image_dims[2]) {
		try {
			// Define an instance name for the session
			std::string instance_name = "inference-session";

			// Create an ONNX Runtime environment with a given logging level
			ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, instance_name.c_str(), &env);

			// Disable telemetry events
			ort->DisableTelemetryEvents(env);

			// Create session options for further configuration
			ort->CreateSessionOptions(&session_options);

			// Define the execution provider
			std::string provider_name = execution_provider;

			// Map execution providers to specific actions (e.g., settings for DML)
			std::unordered_map<std::string, std::function<void()>> execution_provider_actions = {
				{"CPU", []() {}},  // No special settings for CPU
				{"Dml", [&]() {   // Settings for DirectML (DML)
					ort->DisableMemPattern(session_options);
					ort->SetSessionExecutionMode(session_options, ExecutionMode::ORT_SEQUENTIAL);
					OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);
				}}
			};

			// Apply the settings based on the chosen execution provider
			bool action_taken = false;
			for (const auto& pair : execution_provider_actions) {
				const auto& key = pair.first;
				const auto& action = pair.second;

				if (provider_name.find(key) != std::string::npos) {
					action();
					action_taken = true;
					break;
				}
			}

			if (!action_taken) {
				return "Unknown execution provider specified.";
			}

			// Load the ONNX model
			ort->CreateSession(env, stringToWstring(model_path).c_str(), session_options, &session);
			ort->ReleaseSessionOptions(session_options);

			// Set up an allocator for retrieving input-output names
			Ort::AllocatorWithDefaultOptions allocator;

			char* temp_input_name;
			ort->SessionGetInputName(session, 0, allocator, &temp_input_name);
			input_name = temp_input_name;

			char* temp_output_name;
			ort->SessionGetOutputName(session, 0, allocator, &temp_output_name);
			output_name = temp_output_name;

			// Store image dimensions and prepare the input data container
			input_w = image_dims[0];
			input_h = image_dims[1];
			n_pixels = input_w * input_h;
			input_data.resize(n_pixels * n_channels);

			return "Model loaded successfully.";
		}
		catch (const std::exception& e) {
			// Handle standard exceptions and return their messages
			return e.what();
		}
		catch (...) {
			// Handle all other exceptions
			return "An unknown error occurred while loading the model.";
		}
	}

	/// <summary>
	/// Perform inference using the loaded ONNX model.
	/// </summary>
	/// <param name="image_data">Raw image data as bytes.</param>
	/// <param name="output_array">Array to store the inferred results.</param>
	/// <param name="length">Length of the output_array.</param>
	/// <returns></returns>
	DLLExport void PerformInference(byte* image_data, float* output_array, int length) {

		// Preprocessing: Normalize and restructure the image data
		for (int p = 0; p < n_pixels; p++) {
			for (int ch = 0; ch < n_channels; ch++) {
				// Normalize pixel values to [0, 1] and reorder channels
				input_data[ch * n_pixels + p] = (image_data[p * n_channels + ch] / 255.0f);
			}
		}

		// Define the names of input and output tensors for inference
		const char* input_names[] = { input_name.c_str() };
		const char* output_names[] = { output_name.c_str() };

		// Define the shape of the input tensor
		int64_t input_shape[] = { 1, 3, input_h, input_w };

		// Create a memory info instance for CPU allocation
		OrtMemoryInfo* memory_info;
		ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

		// Convert the processed image data into an ONNX tensor format
		OrtValue* input_tensor = nullptr;
		ort->CreateTensorWithDataAsOrtValue(
			memory_info, input_data.data(), input_data.size() * sizeof(float),
			input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
		);

		// Free the memory info after usage
		ort->ReleaseMemoryInfo(memory_info);

		// Perform inference using the ONNX Runtime
		OrtValue* output_tensor = nullptr;
		ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);

		// If inference fails, release resources and return
		if (!output_tensor) {
			ort->ReleaseValue(input_tensor);
			return;
		}

		// Extract data from the output tensor
		float* out_data;
		ort->GetTensorMutableData(output_tensor, (void**)&out_data);

		// Copy the inference results to the provided output array
		std::memcpy(output_array, out_data, length * sizeof(float));

		// Release resources associated with the tensors
		ort->ReleaseValue(input_tensor);
		ort->ReleaseValue(output_tensor);
	}
}
