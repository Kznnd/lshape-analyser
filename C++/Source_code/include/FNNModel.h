#pragma once

#undef slots
#include <torch/torch.h>
#define slots Q_SLOTS

class FNNModelImpl : public torch::nn::Module {
public:
	FNNModelImpl(int input_size, int hidden_size, int output_size)
		: input_fc(input_size, hidden_size),
		hidden_fc(hidden_size, hidden_size),
		output_fc(hidden_size, output_size) {
		register_module("input_fc", input_fc);
		register_module("hidden_fc", hidden_fc);
		register_module("output_fc", output_fc);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(input_fc->forward(x));
		x = torch::relu(hidden_fc->forward(x));
		x = output_fc->forward(x);

		return x;
	}

private:
	torch::nn::Linear input_fc;
	torch::nn::Linear hidden_fc;
	torch::nn::Linear output_fc;
};
TORCH_MODULE(FNNModel);