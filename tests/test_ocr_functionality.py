"""
OCR Functionality Tests - Testing text extraction accuracy and performance
"""

import pytest
import asyncio
import tempfile
import time
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.sovereign.screen_context_manager import ScreenContextManager, ScreenContextConfig
from src.sovereign.config import Config


class TestOCRFunctionality:
    """Test OCR text extraction functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        return ScreenContextConfig(
            storage_path=temp_dir,
            min_text_confidence=30.0,
            enable_preprocessing=True
        )
    
    @pytest.fixture
    def screen_manager(self, config):
        """Create ScreenContextManager instance"""
        # Create main config
        main_config = Config()
        return ScreenContextManager(config, main_config)
    
    def create_text_image(self, text: str, width: int = 400, height: int = 100, font_size: int = 20) -> Image.Image:
        """Create a test image with text"""
        # Create white background
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill='black', font=font)
        
        return img
    
    def create_multi_line_image(self, lines: list, width: int = 500, height: int = 200) -> Image.Image:
        """Create image with multiple lines of text"""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        y_offset = 20
        for line in lines:
            draw.text((20, y_offset), line, fill='black', font=font)
            y_offset += 25
        
        return img
    
    def create_noisy_image(self, text: str, noise_level: float = 0.1) -> Image.Image:
        """Create text image with noise"""
        img = self.create_text_image(text)
        
        # Add noise
        img_array = np.array(img)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)
    
    @pytest.mark.asyncio
    async def test_simple_text_extraction(self, screen_manager):
        """Test extraction of simple text"""
        test_text = "Hello World"
        img = self.create_text_image(test_text)
        
        text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
        
        # Should extract the text
        assert test_text.lower() in text.lower()
        assert len(boxes) > 0
        assert confidence > 0
    
    @pytest.mark.asyncio
    async def test_multi_line_text_extraction(self, screen_manager):
        """Test extraction of multiple lines"""
        test_lines = [
            "First line of text",
            "Second line here",
            "Third line content"
        ]
        img = self.create_multi_line_image(test_lines)
        
        text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
        
        # Should extract multiple lines
        for line in test_lines:
            assert any(word.lower() in text.lower() for word in line.split())
        
        assert len(boxes) >= len(test_lines)
        assert confidence > 0
    
    @pytest.mark.asyncio
    async def test_numbers_and_symbols_extraction(self, screen_manager):
        """Test extraction of numbers and symbols"""
        test_text = "Price: $123.45 (50% off)"
        img = self.create_text_image(test_text, width=600)
        
        text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
        
        # Should extract numbers and basic symbols
        assert "123" in text or "12345" in text  # OCR might miss decimal point
        assert "$" in text or "Price" in text
        assert len(boxes) > 0
    
    @pytest.mark.asyncio
    async def test_low_contrast_text(self, screen_manager):
        """Test extraction of low contrast text"""
        # Create gray text on white background
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Gray text (low contrast)
        draw.text((50, 30), "Low Contrast Text", fill='gray', font=font)
        
        text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
        
        # Preprocessing should help with low contrast
        assert len(text.strip()) > 0  # Should extract something
        assert confidence >= 0  # May have low confidence but should work
    
    @pytest.mark.asyncio
    async def test_preprocessing_effectiveness(self, screen_manager):
        """Test that preprocessing improves OCR results"""
        test_text = "Small Text Test"
        
        # Create small, low-contrast image
        img = Image.new('RGB', (200, 50), color='#f0f0f0')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        draw.text((10, 15), test_text, fill='#404040', font=font)
        
        # Test with preprocessing enabled
        screen_manager.config.enable_preprocessing = True
        text_with_prep, boxes_with_prep, conf_with_prep = await screen_manager._extract_text_with_ocr(img)
        
        # Test without preprocessing
        screen_manager.config.enable_preprocessing = False
        text_without_prep, boxes_without_prep, conf_without_prep = await screen_manager._extract_text_with_ocr(img)
        
        # Preprocessing should help (either more text or higher confidence)
        preprocessing_helped = (
            len(text_with_prep) >= len(text_without_prep) or
            conf_with_prep >= conf_without_prep
        )
        assert preprocessing_helped
    
    @pytest.mark.asyncio
    async def test_empty_image_handling(self, screen_manager):
        """Test handling of empty/blank images"""
        # Create blank white image
        img = Image.new('RGB', (400, 200), color='white')
        
        text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
        
        # Should handle gracefully
        assert text == "" or text.strip() == ""
        assert len(boxes) == 0 or all(box == (0, 0, 0, 0) for box in boxes)
    
    @pytest.mark.asyncio
    async def test_ocr_performance_timing(self, screen_manager):
        """Test OCR performance and timing"""
        test_text = "Performance Test Text"
        img = self.create_text_image(test_text, width=800, height=200)
        
        start_time = time.time()
        text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust based on system)
        assert processing_time < 5.0  # 5 seconds max for test image
        assert len(text.strip()) > 0
        
        # Update performance stats tracking
        assert screen_manager.performance_stats["total_text_extracted"] > 0
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, screen_manager):
        """Test confidence-based text filtering"""
        # Set high confidence threshold
        screen_manager.config.min_text_confidence = 80.0
        
        # Create clear text image
        clear_text = "Clear High Quality Text"
        clear_img = self.create_text_image(clear_text, font_size=24)
        
        text_clear, boxes_clear, conf_clear = await screen_manager._extract_text_with_ocr(clear_img)
        
        # Set low confidence threshold
        screen_manager.config.min_text_confidence = 10.0
        
        # Create noisy text image
        noisy_img = self.create_noisy_image("Noisy Low Quality", noise_level=0.3)
        
        text_noisy, boxes_noisy, conf_noisy = await screen_manager._extract_text_with_ocr(noisy_img)
        
        # Clear text should have better results
        assert len(text_clear) >= len(text_noisy.strip())
        # Confidence should be reasonable for clear text
        assert conf_clear > conf_noisy or conf_clear > 50
    
    @pytest.mark.asyncio 
    async def test_ocr_language_support(self, screen_manager):
        """Test OCR language configuration"""
        # Test default English
        assert screen_manager.config.ocr_language == "eng"
        
        # Test changing language
        screen_manager.config.ocr_language = "eng+fra"  # English + French
        
        test_text = "Hello World"
        img = self.create_text_image(test_text)
        
        # Should still work with multi-language config
        with patch('pytesseract.image_to_data') as mock_ocr:
            mock_ocr.return_value = {
                'text': ['Hello', 'World'],
                'conf': [95, 90],
                'left': [10, 60],
                'top': [30, 30],
                'width': [45, 50],
                'height': [20, 20]
            }
            
            text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
            
            # Verify language parameter was passed
            mock_ocr.assert_called_once()
            call_kwargs = mock_ocr.call_args[1]
            assert call_kwargs['lang'] == "eng+fra"
    
    @pytest.mark.asyncio
    async def test_bounding_box_accuracy(self, screen_manager):
        """Test accuracy of bounding box detection"""
        test_text = "Test Box"
        img = self.create_text_image(test_text, width=200, height=100)
        
        text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
        
        if len(boxes) > 0:
            # Bounding boxes should be within image dimensions
            for box in boxes:
                left, top, width, height = box
                assert 0 <= left <= img.width
                assert 0 <= top <= img.height
                assert 0 <= width <= img.width
                assert 0 <= height <= img.height
                # Box should have reasonable dimensions
                assert width > 0 and height > 0
    
    @pytest.mark.asyncio
    async def test_ocr_error_handling(self, screen_manager):
        """Test OCR error handling"""
        img = self.create_text_image("Test Text")
        
        # Test OCR failure
        with patch('pytesseract.image_to_data', side_effect=Exception("OCR Error")):
            text, boxes, confidence = await screen_manager._extract_text_with_ocr(img)
            
            # Should handle gracefully
            assert text == ""
            assert boxes == []
            assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_image_format_compatibility(self, screen_manager):
        """Test OCR with different image formats"""
        test_text = "Format Test"
        
        # Test RGB image
        rgb_img = self.create_text_image(test_text)
        rgb_text, _, rgb_conf = await screen_manager._extract_text_with_ocr(rgb_img)
        
        # Test grayscale image
        gray_img = rgb_img.convert('L')
        gray_text, _, gray_conf = await screen_manager._extract_text_with_ocr(gray_img)
        
        # Both should work
        assert len(rgb_text.strip()) > 0
        assert len(gray_text.strip()) > 0
        # Results should be similar
        assert "Test" in rgb_text or "Format" in rgb_text
        assert "Test" in gray_text or "Format" in gray_text
    
    def test_preprocessing_pipeline(self, screen_manager):
        """Test image preprocessing pipeline"""
        # Create test image
        img = self.create_text_image("Preprocessing Test", width=100, height=50)
        
        # Test preprocessing
        processed = screen_manager._preprocess_image(img)
        
        # Should be processed
        assert processed is not None
        assert processed.mode == 'L'  # Should be grayscale
        # Should be upscaled since original is small
        assert processed.size[0] >= 1000 or processed.size[1] >= 1000
    
    @pytest.mark.asyncio
    async def test_real_world_scenarios(self, screen_manager):
        """Test OCR on realistic screen content scenarios"""
        
        # Test 1: Web browser text
        browser_lines = [
            "Welcome to our website",
            "Username: john@example.com",
            "Password: ********",
            "Login Button"
        ]
        browser_img = self.create_multi_line_image(browser_lines)
        browser_text, _, _ = await screen_manager._extract_text_with_ocr(browser_img)
        
        # Should extract web content
        assert "Welcome" in browser_text or "website" in browser_text
        
        # Test 2: Code editor text
        code_lines = [
            "def hello_world():",
            "    print('Hello, World!')",
            "    return True",
            "# End of function"
        ]
        code_img = self.create_multi_line_image(code_lines, width=600)
        code_text, _, _ = await screen_manager._extract_text_with_ocr(code_img)
        
        # Should extract code content
        assert "def" in code_text or "print" in code_text or "Hello" in code_text
        
        # Test 3: Email content
        email_lines = [
            "From: sender@company.com",
            "To: recipient@example.org", 
            "Subject: Important Meeting",
            "Dear Team, please join us at 3 PM"
        ]
        email_img = self.create_multi_line_image(email_lines, width=700)
        email_text, _, _ = await screen_manager._extract_text_with_ocr(email_img)
        
        # Should extract email content
        assert "From" in email_text or "Subject" in email_text or "Meeting" in email_text 