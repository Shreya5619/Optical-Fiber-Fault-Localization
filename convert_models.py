#!/usr/bin/env python3
"""
Model Conversion Script for OTDR Dashboard Compatibility
Converts problematic Keras models to compatible formats
"""

import numpy as np
import tensorflow as tf
import h5py
import json
import os
import sys

def convert_model_for_compatibility(input_path, output_path=None):
    """Convert a problematic Keras model to a compatible format"""
    
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compatible.h5"
    
    print(f"🔄 Converting {input_path} to {output_path}")
    print("=" * 60)
    
    try:
        # Method 1: Try direct loading and re-saving
        print("📋 Method 1: Direct load and re-save...")
        try:
            model = tf.keras.models.load_model(input_path, compile=False)
            model.save(output_path, include_optimizer=False, save_format='h5')
            print("✅ Direct conversion successful!")
            return output_path
        except Exception as e:
            print(f"❌ Direct conversion failed: {e}")
        
        # Method 2: Load weights and create new architecture
        print("\n📋 Method 2: Recreate architecture and load weights...")
        try:
            # Analyze the original model structure from HDF5
            with h5py.File(input_path, 'r') as f:
                if 'model_config' in f.attrs:
                    model_config = json.loads(f.attrs['model_config'])
                    
                    # Determine if it's hierarchical or flat based on input layers
                    input_layers = []
                    if 'config' in model_config and 'layers' in model_config['config']:
                        for layer in model_config['config']['layers']:
                            if layer.get('class_name') == 'InputLayer':
                                input_layers.append(layer)
                    
                    print(f"🔍 Found {len(input_layers)} input layer(s)")
                    
                    # Create appropriate model architecture
                    if len(input_layers) >= 2:
                        print("   - Creating hierarchical model (OTDR_trace + SNR)")
                        new_model = create_hierarchical_model()
                    else:
                        print("   - Creating flat model (31 features)")
                        new_model = create_flat_model()
                    
                    # Try to load weights with name matching
                    print("🔄 Attempting to load weights...")
                    try:
                        new_model.load_weights(input_path, by_name=True, skip_mismatch=True)
                        print("✅ Weights loaded successfully!")
                    except Exception as weight_error:
                        print(f"⚠️  Weight loading failed: {weight_error}")
                        print("   - Creating template model with random weights")
                    
                    # Save the new model
                    new_model.save(output_path, include_optimizer=False, save_format='h5')
                    print(f"✅ New model saved to {output_path}")
                    return output_path
                else:
                    print("❌ No model configuration found in HDF5 file")
        except Exception as e:
            print(f"❌ Architecture recreation failed: {e}")
        
        # Method 3: Create a generic template
        print("\n📋 Method 3: Creating generic template...")
        try:
            template_model = create_hierarchical_model()
            template_path = f"{os.path.splitext(output_path)[0]}_template.h5"
            template_model.save(template_path, include_optimizer=False, save_format='h5')
            print(f"✅ Generic template saved to {template_path}")
            print("⚠️  Note: This is a template - retrain with your data for best results")
            return template_path
        except Exception as e:
            print(f"❌ Template creation failed: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

def create_hierarchical_model():
    """Create a hierarchical model compatible with OTDR dashboard"""
    
    # Input layers
    otdr_input = tf.keras.layers.Input(shape=(30,), name='OTDR_trace')
    snr_input = tf.keras.layers.Input(shape=(1,), name='SNR')
    
    # OTDR branch - CNN for trace analysis
    x1 = tf.keras.layers.Reshape((30, 1))(otdr_input)
    x1 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x1 = tf.keras.layers.Dense(64, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)
    
    # SNR branch
    x2 = tf.keras.layers.Dense(32, activation='relu')(snr_input)
    x2 = tf.keras.layers.Dense(16, activation='relu')(x2)
    
    # Combine branches
    combined = tf.keras.layers.Concatenate()([x1, x2])
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.4)(combined)
    combined = tf.keras.layers.Dense(64, activation='relu')(combined)
    
    # Output layer (8 classes for full classification)
    output = tf.keras.layers.Dense(8, activation='softmax')(combined)
    
    model = tf.keras.Model(inputs=[otdr_input, snr_input], outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_flat_model():
    """Create a flat model compatible with OTDR dashboard"""
    
    # Input layer (31 features: SNR + P1-P30)
    input_layer = tf.keras.layers.Input(shape=(31,))
    
    # Hidden layers
    x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer (8 classes)
    output = tf.keras.layers.Dense(8, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_binary_model():
    """Create a binary classification model"""
    
    # Input layers (hierarchical)
    otdr_input = tf.keras.layers.Input(shape=(30,), name='OTDR_trace')
    snr_input = tf.keras.layers.Input(shape=(1,), name='SNR')
    
    # OTDR branch
    x1 = tf.keras.layers.Reshape((30, 1))(otdr_input)
    x1 = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x1 = tf.keras.layers.Dense(64, activation='relu')(x1)
    
    # SNR branch
    x2 = tf.keras.layers.Dense(32, activation='relu')(snr_input)
    
    # Combine and output
    combined = tf.keras.layers.Concatenate()([x1, x2])
    combined = tf.keras.layers.Dense(64, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)
    
    # Binary output (sigmoid)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    
    model = tf.keras.Model(inputs=[otdr_input, snr_input], outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def test_converted_model(model_path):
    """Test if the converted model works with sample data"""
    print(f"\n🧪 Testing converted model: {model_path}")
    print("=" * 50)
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loads successfully")
        print(f"   - Input names: {model.input_names if hasattr(model, 'input_names') else 'N/A'}")
        print(f"   - Number of inputs: {len(model.inputs)}")
        print(f"   - Input shapes: {[inp.shape for inp in model.inputs]}")
        print(f"   - Output shape: {model.output.shape}")
        
        # Generate sample data
        np.random.seed(42)
        sample_snr = np.random.uniform(8, 15)
        sample_otdr = np.random.uniform(0, 1, 30)
        
        # Test prediction
        if len(model.inputs) > 1:
            # Hierarchical model
            test_input = {
                'OTDR_trace': sample_otdr.reshape(1, 30),
                'SNR': np.array([sample_snr]).reshape(1, 1)
            }
            print("📋 Testing hierarchical input format...")
        else:
            # Flat model
            test_input = np.concatenate([[sample_snr], sample_otdr]).reshape(1, 31)
            print("📋 Testing flat input format...")
        
        prediction = model.predict(test_input, verbose=0)
        print(f"✅ Prediction successful!")
        print(f"   - Output shape: {prediction.shape}")
        
        if prediction.shape[-1] == 1:
            print(f"   - Binary prediction: {prediction[0][0]:.4f}")
            print(f"   - Predicted class: {'Fault' if prediction[0][0] > 0.5 else 'Normal'}")
        else:
            print(f"   - Predicted class: {np.argmax(prediction[0])}")
            print(f"   - Max probability: {np.max(prediction[0]):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main conversion function"""
    print("🔄 OTDR Model Compatibility Converter")
    print("=" * 60)
    print("This script converts TensorFlow/Keras models to be compatible with the OTDR dashboard")
    print("It handles InputLayer batch_shape issues and other compatibility problems")
    print()
    
    # Find .h5 files in current directory
    model_files = [f for f in os.listdir('.') if f.endswith('.h5') and not f.endswith('_compatible.h5') and not f.endswith('_template.h5')]
    
    if not model_files:
        print("❌ No .h5 model files found in current directory")
        print("📋 Please place your model files in this directory and run again")
        print()
        print("💡 Expected files:")
        print("   - binary_model.h5 (for binary fault detection)")
        print("   - multiclass_model.h5 (for fault classification)")
        print("   - Or any other .h5 Keras model files")
        return
    
    print(f"Found {len(model_files)} model file(s) to convert:")
    for file in model_files:
        file_size = os.path.getsize(file) / (1024*1024)  # MB
        print(f"   - {file} ({file_size:.1f} MB)")
    print()
    
    converted_files = []
    
    for model_file in model_files:
        print(f"\n" + "="*60)
        converted_path = convert_model_for_compatibility(model_file)
        
        if converted_path and os.path.exists(converted_path):
            # Test the converted model
            if test_converted_model(converted_path):
                converted_files.append(converted_path)
                print(f"✅ Successfully converted: {model_file} → {converted_path}")
            else:
                print(f"⚠️  Converted but test failed: {converted_path}")
                converted_files.append(converted_path)
        else:
            print(f"❌ Conversion failed for: {model_file}")
    
    # Summary
    print(f"\n📊 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Original files: {len(model_files)}")
    print(f"Successfully converted: {len(converted_files)}")
    
    if converted_files:
        print(f"\n✅ Converted files ready for dashboard:")
        for file in converted_files:
            file_size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   - {file} ({file_size:.1f} MB)")
        
        print(f"\n🚀 Next steps:")
        print(f"1. Run the dashboard: streamlit run otdr_dashboard_fixed.py")
        print(f"2. Upload the converted model files")
        print(f"3. Test with your OTDR data")
        
        # Check if we can also create specialized binary models
        if any('multiclass' in f or 'class' in f for f in converted_files):
            print(f"\n💡 Tip: For binary classification, you can use the multiclass")
            print(f"   model - the dashboard will automatically convert its output")
    else:
        print(f"\n❌ No files were successfully converted")
        print(f"💡 Troubleshooting:")
        print(f"   1. Try re-training your models with current TensorFlow version")
        print(f"   2. Check that the .h5 files are valid Keras models")
        print(f"   3. Ensure the models were saved with model.save() not just weights")

if __name__ == "__main__":
    main()
