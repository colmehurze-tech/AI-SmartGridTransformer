import torch.onnx
from init_script import model

model.eval()
dummy_input = torch.randn(1, 60, 2) 

onnx_file_path = "smart_grid_forecast.onnx"

torch.onnx.export(
    model,dummy_input,onnx_file_path,export_params=True,opset_version=17,          
    do_constant_folding=True,  
    input_names=['input'],     
    output_names=['output'],  
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model exported to {onnx_file_path}.")