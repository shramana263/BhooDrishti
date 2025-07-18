"""
Advanced Model Testing Script for BhooDrishti
=============================================

Comprehensive testing script for the advanced deep learning models.
Imports processor, loads test images, and demonstrates AI capabilities.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our advanced modules
try:
    from advanced_dl_processor import AdvancedDLProcessor, process_images_with_advanced_ai
    from advanced_dl_models import create_advanced_system, preprocess_png_for_dl
    print("✅ Successfully imported advanced AI modules")
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing advanced modules: {e}")
    ADVANCED_AVAILABLE = False
    sys.exit(1)

class AdvancedModelTester:
    """
    Comprehensive tester for advanced change detection models
    """
    
    def __init__(self, test_data_dir=None):
        self.test_data_dir = Path(test_data_dir) if test_data_dir else Path("test_image")
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        print("🤖 Initializing Advanced AI Processor...")
        self.processor = AdvancedDLProcessor()
        print(f"✅ Processor initialized on {self.processor.device}")
        
    def create_sample_images(self):
        """Create sample test images if none exist"""
        print("🎨 Creating sample test images...")
        
        # Create test images directory
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic satellite-like images
        img_size = (512, 512, 3)
        
        # Image 1: Base landscape
        img1 = np.random.randint(80, 150, img_size, dtype=np.uint8)
        
        # Add some vegetation (green areas)
        vegetation_mask = np.random.random((img_size[0], img_size[1])) > 0.7
        img1[vegetation_mask, 1] = np.random.randint(120, 200, np.sum(vegetation_mask))  # Green channel
        img1[vegetation_mask, 0] = np.random.randint(60, 100, np.sum(vegetation_mask))   # Red channel
        
        # Add water bodies (blue areas)
        water_mask = np.random.random((img_size[0], img_size[1])) > 0.9
        img1[water_mask, 2] = np.random.randint(150, 255, np.sum(water_mask))  # Blue channel
        img1[water_mask, 0] = np.random.randint(30, 80, np.sum(water_mask))    # Red channel
        img1[water_mask, 1] = np.random.randint(50, 120, np.sum(water_mask))   # Green channel
        
        # Image 2: Modified landscape with changes
        img2 = img1.copy()
        
        # Simulate deforestation (remove some vegetation)
        deforest_mask = vegetation_mask & (np.random.random((img_size[0], img_size[1])) > 0.8)
        img2[deforest_mask] = np.random.randint(100, 140, (np.sum(deforest_mask), 3))
        
        # Simulate urban expansion (add gray/white areas)
        urban_mask = np.random.random((img_size[0], img_size[1])) > 0.95
        img2[urban_mask] = np.random.randint(180, 255, (np.sum(urban_mask), 3))
        
        # Add some noise to make it more realistic
        noise1 = np.random.normal(0, 5, img_size).astype(np.int16)
        noise2 = np.random.normal(0, 5, img_size).astype(np.int16)
        
        img1 = np.clip(img1.astype(np.int16) + noise1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)
        
        # Save images
        img1_path = self.test_data_dir / "test_image_1.png"
        img2_path = self.test_data_dir / "test_image_2.png"
        
        cv2.imwrite(str(img1_path), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img2_path), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Sample images created: {img1_path}, {img2_path}")
        return str(img1_path), str(img2_path)
    
    def find_test_images(self):
        """Find existing test images or create samples"""
        print("🔍 Looking for test images...")
        
        # Look for existing images
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
        existing_images = []
        print(self.test_data_dir)
        if self.test_data_dir.exists():
            for ext in image_extensions:
                existing_images.extend(list(self.test_data_dir.glob(f"*{ext}")))
        
        if len(existing_images) >= 2:
            print(f"📁 Found {len(existing_images)} existing images")
            return str(existing_images[0]), str(existing_images[1])
        else:
            print("📁 No sufficient test images found, creating samples...")
            return self.create_sample_images()
    
    def test_basic_inference(self, img1_path, img2_path):
        """Test basic model inference"""
        print("\n🧪 Testing Basic Model Inference...")
        
        try:
            start_time = time.time()
            
            # Load and preprocess images
            tensor1 = preprocess_png_for_dl(img1_path, target_size=(256, 256))
            tensor2 = preprocess_png_for_dl(img2_path, target_size=(256, 256))
            
            # Move to device
            tensor1 = tensor1.to(self.processor.device)
            tensor2 = tensor2.to(self.processor.device)
            
            # Run inference
            self.processor.model.eval()
            with torch.no_grad():
                outputs = self.processor.model(tensor1, tensor2)
            
            inference_time = time.time() - start_time
            
            print(f"✅ Basic inference completed in {inference_time:.2f} seconds")
            print(f"   Change map shape: {outputs['change_map'].shape}")
            print(f"   Quality scores: {outputs['quality1'].item():.3f}, {outputs['quality2'].item():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic inference failed: {e}")
            return False
    
    def test_advanced_detection(self, img1_path, img2_path):
        """Test advanced change detection pipeline"""
        print("\n🔬 Testing Advanced Change Detection Pipeline...")
        
        try:
            start_time = time.time()
            
            # Run advanced detection
            change_results = self.processor.detect_changes_advanced(img1_path, img2_path)
            
            detection_time = time.time() - start_time
            
            print(f"✅ Advanced detection completed in {detection_time:.2f} seconds")
            print(f"   Total pixels analyzed: {change_results['statistics']['total_pixels']:,}")
            print(f"   Changed pixels: {change_results['statistics']['changed_pixels']:,}")
            print(f"   Change percentage: {change_results['statistics']['change_percentage']:.2f}%")
            print(f"   Change area: {change_results['statistics']['change_area_ha']:.2f} hectares")
            
            return change_results
            
        except Exception as e:
            print(f"❌ Advanced detection failed: {e}")
            return None
    
    def test_change_analysis(self, change_results, img1_path, img2_path):
        """Test change type analysis"""
        print("\n📊 Testing Change Type Analysis...")
        
        try:
            start_time = time.time()
            
            # Run change analysis
            analysis_results = self.processor.analyze_change_types_advanced(
                change_results, img1_path, img2_path
            )
            
            analysis_time = time.time() - start_time
            
            print(f"✅ Change analysis completed in {analysis_time:.2f} seconds")
            print(f"   Vegetation changes: {analysis_results['vegetation_changes']['area_ha']:.2f} ha")
            print(f"   Urban changes: {analysis_results['urban_changes']['area_ha']:.2f} ha")
            print(f"   Water changes: {analysis_results['water_changes']['area_ha']:.2f} ha")
            print(f"   Overall confidence: {analysis_results['confidence_scores']['overall_confidence']:.3f}")
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Change analysis failed: {e}")
            return None
    
    def test_visualization(self, change_results, analysis_results, img1_path, img2_path):
        """Test visualization creation"""
        print("\n🎨 Testing Visualization Creation...")
        
        try:
            start_time = time.time()
            
            # Create visualizations
            viz_results = self.processor.create_advanced_visualization(
                change_results, analysis_results, img1_path, img2_path, str(self.output_dir)
            )
            
            viz_time = time.time() - start_time
            
            print(f"✅ Visualization completed in {viz_time:.2f} seconds")
            print(f"   Dashboard: {viz_results['dashboard_path']}")
            print(f"   Change map: {viz_results['change_map_path']}")
            print(f"   Probability map: {viz_results['probability_map_path']}")
            
            return viz_results
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return None
    
    def test_report_export(self, change_results, analysis_results):
        """Test report export"""
        print("\n📋 Testing Report Export...")
        
        try:
            start_time = time.time()
            
            # Export report
            report_path = self.processor.export_advanced_report(
                change_results, analysis_results, 
                str(self.output_dir / "test_report.json")
            )
            
            export_time = time.time() - start_time
            
            print(f"✅ Report export completed in {export_time:.2f} seconds")
            print(f"   Report saved: {report_path}")
            
            # Load and display summary
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            print(f"   Report type: {report_data['metadata']['report_type']}")
            print(f"   Model info: {report_data['metadata']['model_info']['model_type']}")
            
            return report_path
            
        except Exception as e:
            print(f"❌ Report export failed: {e}")
            return None
    
    def test_full_pipeline(self, img1_path, img2_path):
        """Test the complete processing pipeline"""
        print("\n🚀 Testing Complete AI Pipeline...")
        
        try:
            start_time = time.time()
            
            # Run complete pipeline
            result = process_images_with_advanced_ai(
                img1_path, img2_path, str(self.output_dir)
            )
            
            pipeline_time = time.time() - start_time
            
            if result['success']:
                print(f"✅ Complete pipeline completed in {pipeline_time:.2f} seconds")
                print(f"   Device used: {result['processor_info']['device_used']}")
                print(f"   Model type: {result['processor_info']['model_type']}")
                print(f"   Report: {result['report_path']}")
                print(f"   Visualizations: {result['visualizations']['output_directory']}")
                
                return result
            else:
                print(f"❌ Pipeline failed: {result['error']}")
                return None
                
        except Exception as e:
            print(f"❌ Complete pipeline failed: {e}")
            return None
    
    def performance_benchmark(self, img1_path, img2_path, num_runs=3):
        """Run performance benchmark"""
        print(f"\n⚡ Running Performance Benchmark ({num_runs} runs)...")
        
        times = []
        
        for i in range(num_runs):
            print(f"   Run {i+1}/{num_runs}...")
            start_time = time.time()
            
            try:
                change_results = self.processor.detect_changes_advanced(img1_path, img2_path)
                run_time = time.time() - start_time
                times.append(run_time)
                print(f"   ✅ Run {i+1} completed in {run_time:.2f} seconds")
                
            except Exception as e:
                print(f"   ❌ Run {i+1} failed: {e}")
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"\n📈 Performance Summary:")
            print(f"   Average time: {avg_time:.2f} ± {std_time:.2f} seconds")
            print(f"   Min time: {min_time:.2f} seconds")
            print(f"   Max time: {max_time:.2f} seconds")
            print(f"   Device: {self.processor.device}")
            
            return {
                'average_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'num_runs': len(times)
            }
        
        return None
    
    def run_comprehensive_test(self, img1_path=None, img2_path=None):
        """Run comprehensive testing suite"""
        print("🧪 Starting Comprehensive Advanced AI Testing Suite")
        print("=" * 60)
        
        # Find or create test images
        if not img1_path or not img2_path:
            img1_path, img2_path = self.find_test_images()
        
        print(f"📸 Using test images:")
        print(f"   Image 1: {img1_path}")
        print(f"   Image 2: {img2_path}")
        
        # Test suite
        test_results = {}
        
        # 1. Basic inference test
        test_results['basic_inference'] = self.test_basic_inference(img1_path, img2_path)
        
        # 2. Advanced detection test
        change_results = self.test_advanced_detection(img1_path, img2_path)
        test_results['advanced_detection'] = change_results is not None
        
        if change_results:
            # 3. Change analysis test
            analysis_results = self.test_change_analysis(change_results, img1_path, img2_path)
            test_results['change_analysis'] = analysis_results is not None
            
            if analysis_results:
                # 4. Visualization test
                viz_results = self.test_visualization(change_results, analysis_results, img1_path, img2_path)
                test_results['visualization'] = viz_results is not None
                
                # 5. Report export test
                report_path = self.test_report_export(change_results, analysis_results)
                test_results['report_export'] = report_path is not None
        
        # 6. Full pipeline test
        pipeline_result = self.test_full_pipeline(img1_path, img2_path)
        test_results['full_pipeline'] = pipeline_result is not None
        
        # 7. Performance benchmark
        benchmark_results = self.performance_benchmark(img1_path, img2_path)
        test_results['benchmark'] = benchmark_results
        
        # Print summary
        self.print_test_summary(test_results)
        
        return test_results
    
    def print_test_summary(self, test_results):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for k, v in test_results.items() if k != 'benchmark' and v)
        total = len(test_results) - 1  # Exclude benchmark
        
        print(f"Overall Status: {passed}/{total} tests passed")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print()
        
        # Individual test results
        test_names = {
            'basic_inference': 'Basic Model Inference',
            'advanced_detection': 'Advanced Change Detection',
            'change_analysis': 'Change Type Analysis',
            'visualization': 'Visualization Creation',
            'report_export': 'Report Export',
            'full_pipeline': 'Complete Pipeline'
        }
        
        for key, name in test_names.items():
            status = "✅ PASS" if test_results.get(key, False) else "❌ FAIL"
            print(f"{name}: {status}")
        
        # Performance summary
        if test_results.get('benchmark'):
            benchmark = test_results['benchmark']
            print(f"\nPerformance Benchmark:")
            print(f"  Average Processing Time: {benchmark['average_time']:.2f}s")
            print(f"  Device Used: {self.processor.device}")
        
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"Generated Files:")
        
        # List generated files
        if self.output_dir.exists():
            for file_path in self.output_dir.glob("*"):
                if file_path.is_file():
                    print(f"  - {file_path.name}")

def main():
    """Main testing function"""
    print("🤖 BhooDrishti Advanced AI Model Tester")
    print("=======================================")
    
    # Check if advanced AI is available
    if not ADVANCED_AVAILABLE:
        print("❌ Advanced AI modules not available. Please install required dependencies.")
        sys.exit(1)
    
    # Initialize tester
    tester = AdvancedModelTester('C:/Users/param/Core/Code/Hackathon/BhooDrishti/assets/test_image')
    
    # Run comprehensive test
    test_results = tester.run_comprehensive_test()
    
    print("\n🎉 Testing completed!")
    print(f"Check output directory: {tester.output_dir}")

if __name__ == "__main__":
    main()
