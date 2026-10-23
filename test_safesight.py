import unittest

import numpy as np
from PIL import Image, ImageFilter

from safesight import AssessView, DegradeView


class SafeSightTests(unittest.TestCase):
    def setUp(self):
        rows, columns = np.indices((240, 320))
        checks = (rows // 8 + columns // 8) % 2
        texture = (65 + checks * 125).astype(np.uint8)
        self.image = np.repeat(texture[:, :, None], 3, axis=2)

    def test_TexturedMidrangeImageHasNoMeasuredIssue(self):
        assessment = AssessView(self.image)
        self.assertEqual(assessment["status"], "no_issue")
        self.assertEqual(assessment["issues"], ())
        self.assertGreater(assessment["detail"], 0)
        self.assertGreater(assessment["mean_luminance"], 0.4)
        self.assertLess(assessment["mean_luminance"], 0.6)
        self.assertEqual(assessment["dark_fraction"], 0)
        self.assertEqual(assessment["bright_fraction"], 0)

    def test_DimmingProducesADarkAdvisory(self):
        dark = (self.image.astype(float) * 0.1).astype(np.uint8)
        assessment = AssessView(dark)
        self.assertEqual(assessment["status"], "limited")
        self.assertIn("dark", assessment["issues"])
        self.assertNotIn("bright", assessment["issues"])
        self.assertLess(assessment["mean_luminance"],
                        AssessView(self.image)["mean_luminance"])

    def test_WashedOutImageProducesABrightAdvisory(self):
        bright = (self.image.astype(float) * 0.1 + 230).astype(np.uint8)
        assessment = AssessView(bright)
        self.assertEqual(assessment["status"], "limited")
        self.assertIn("bright", assessment["issues"])
        self.assertNotIn("dark", assessment["issues"])
        self.assertGreater(assessment["mean_luminance"],
                           AssessView(self.image)["mean_luminance"])

    def test_BlurringReducesDetailWithoutClaimingACause(self):
        blurred = np.asarray(Image.fromarray(self.image).filter(ImageFilter.GaussianBlur(8)))
        assessment = AssessView(blurred)
        self.assertEqual(assessment["status"], "limited")
        self.assertEqual(assessment["issues"], ("low_detail",))
        self.assertLess(assessment["detail"], AssessView(self.image)["detail"] / 10)

    def test_UniformMidrangeSurfaceIsAlsoLowDetail(self):
        assessment = AssessView(np.full((240, 320, 3), 128, dtype=np.uint8))
        self.assertEqual(assessment["issues"], ("low_detail",))
        self.assertEqual(assessment["status"], "limited")
        self.assertAlmostEqual(assessment["detail"], 0, places=6)

    def test_LargeDarkAreaIsReportedDespiteModerateMean(self):
        image = np.zeros_like(self.image)
        image[:, :96] = 255
        assessment = AssessView(image)
        self.assertGreater(assessment["mean_luminance"], 0.2)
        self.assertIn("dark", assessment["issues"])

    def test_LargeSaturatedAreaIsReportedDespiteModerateMean(self):
        image = np.zeros_like(self.image)
        image[:, :192] = 255
        assessment = AssessView(image)
        self.assertLess(assessment["mean_luminance"], 0.7)
        self.assertIn("bright", assessment["issues"])

    def test_SmallShadowsAndHighlightsDoNotSetExposureWarnings(self):
        image = self.image.copy()
        image[:24] = 255
        image[-24:] = 0
        assessment = AssessView(image)
        self.assertNotIn("dark", assessment["issues"])
        self.assertNotIn("bright", assessment["issues"])

    def test_EquivalentLargerFrameRetainsTheSameAdvisory(self):
        larger = np.asarray(Image.fromarray(self.image).resize((1280, 960),
                                                              Image.Resampling.NEAREST))
        original = AssessView(self.image)
        resized = AssessView(larger)
        self.assertEqual(resized["issues"], original["issues"])
        self.assertEqual(resized["status"], original["status"])
        self.assertAlmostEqual(resized["mean_luminance"], original["mean_luminance"],
                               delta=0.01)

    def test_MinimumSizedAndExtremeAspectFramesHaveFiniteMeasurements(self):
        for shape in [(16, 16, 3), (16, 10000, 3), (10000, 16, 3)]:
            with self.subTest(shape=shape):
                assessment = AssessView(np.full(shape, 128, dtype=np.uint8))
                self.assertEqual(assessment["issues"], ("low_detail",))
                for measurement in ["mean_luminance", "dark_fraction", "bright_fraction", "detail"]:
                    self.assertTrue(np.isfinite(assessment[measurement]))
                    self.assertGreaterEqual(assessment[measurement], 0)

    def test_ReadonlyNoncontiguousInputIsAcceptedAndUnchanged(self):
        image = self.image[:, ::-1]
        image.flags.writeable = False
        original = image.copy()
        assessment = AssessView(image)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(assessment["status"], "no_issue")
        self.assertIsInstance(assessment["issues"], tuple)
        for measurement in ["mean_luminance", "dark_fraction", "bright_fraction", "detail"]:
            self.assertIsInstance(assessment[measurement], float)

    def test_InvalidInputTypesShapesAndPixelFormatsAreRejected(self):
        invalid = [None, [], self.image.tolist(), self.image.astype(float),
                   self.image.astype(np.int16), np.zeros((240, 320), dtype=np.uint8),
                   np.zeros((240, 320, 4), dtype=np.uint8),
                   np.zeros((240, 320, 1), dtype=np.uint8),
                   np.zeros((15, 320, 3), dtype=np.uint8),
                   np.zeros((240, 0, 3), dtype=np.uint8),
                   np.zeros((1, 240, 320, 3), dtype=np.uint8)]
        for image in invalid:
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                AssessView(image)

    def test_ZeroEffectsReturnAnExactIndependentCopy(self):
        output = DegradeView(self.image)
        np.testing.assert_array_equal(output, self.image)
        self.assertFalse(np.shares_memory(output, self.image))
        output[:] = 0
        self.assertGreater(self.image.mean(), 0)

    def test_EffectsPreserveDimensionsAndRGBPixelFormat(self):
        for shape in [(16, 16, 3), (96, 160, 3), (160, 96, 3)]:
            with self.subTest(shape=shape):
                image = np.full(shape, 128, dtype=np.uint8)
                output = DegradeView(image, blur=0.5, exposure=-0.5, haze=0.25)
                self.assertEqual(output.shape, image.shape)
                self.assertEqual(output.dtype, np.uint8)

    def test_EffectsAreDeterministicAndDoNotMutateReadonlyNoncontiguousInput(self):
        image = self.image[:, ::-1]
        image.flags.writeable = False
        original = image.copy()
        first = DegradeView(image, blur=0.4, exposure=-0.5, haze=0.25)
        second = DegradeView(image, blur=0.4, exposure=-0.5, haze=0.25)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(first, image))

    def test_MaximumBlurReducesMeasuredDetail(self):
        output = DegradeView(self.image, blur=1)
        assessment = AssessView(output)
        self.assertEqual(assessment["issues"], ("low_detail",))
        self.assertLess(assessment["detail"], AssessView(self.image)["detail"] / 10)

    def test_ExposureEndpointsProduceDarkAndBrightAdvisories(self):
        for exposure, issue in [(-4, "dark"), (4, "bright")]:
            with self.subTest(exposure=exposure):
                output = DegradeView(self.image, exposure=exposure)
                self.assertIn(issue, AssessView(output)["issues"])

    def test_ExposureDoublesLinearLightRatherThanEncodedPixelValues(self):
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        brighter = DegradeView(image, exposure=1)
        darker = DegradeView(image, exposure=-1)
        np.testing.assert_array_equal(brighter, np.full_like(image, 176))
        np.testing.assert_array_equal(darker, np.full_like(image, 92))
        self.assertLess(brighter.max(), 255)

    def test_FullHazeIsNeutralLowDetailAfterAllOtherEffects(self):
        for exposure in [-4, 0, 4]:
            with self.subTest(exposure=exposure):
                output = DegradeView(self.image, blur=1, exposure=exposure, haze=1)
                np.testing.assert_array_equal(output,
                                             np.broadcast_to([185, 190, 195], self.image.shape))
                self.assertEqual(AssessView(output)["issues"], ("low_detail",))

    def test_HazeReducesContrastAndBlendsTheSpecifiedNeutralColour(self):
        output = DegradeView(self.image, haze=0.5)
        expected = np.rint(self.image.astype(float) * 0.5
                           + np.array([185, 190, 195]) * 0.5).astype(np.uint8)
        np.testing.assert_allclose(output, expected, atol=1, rtol=0)
        self.assertLess(output.std(), self.image.std())

    def test_BlurScalesWithWidthForEquivalentLargerFrames(self):
        larger = np.asarray(Image.fromarray(self.image).resize((1280, 960),
                                                              Image.Resampling.NEAREST))
        low = DegradeView(self.image, blur=0.7)
        high = DegradeView(larger, blur=0.7)
        reduced = np.asarray(Image.fromarray(high).resize((320, 240),
                                                          Image.Resampling.LANCZOS))
        self.assertLess(np.abs(low.astype(float) - reduced).mean(), 1)
        self.assertEqual(AssessView(low)["issues"], AssessView(high)["issues"])

    def test_InvalidEffectsAreRejectedEvenWhenOtherEffectsAreZero(self):
        for argument, invalid in [
                ("blur", [-0.1, 1.1, np.nan, np.inf, -np.inf, [0.5], True, "invalid"]),
                ("exposure", [-4.1, 4.1, np.nan, np.inf, -np.inf, [0], False, None]),
                ("haze", [-0.1, 1.1, np.nan, np.inf, -np.inf, [0.5], True, "invalid"])]:
            for value in invalid:
                with self.subTest(argument=argument, value=value), self.assertRaises(ValueError):
                    DegradeView(self.image, **{argument: value})

    def test_DegradationRejectsInvalidImagesBeforeIdentityCopy(self):
        for image in [None, self.image.tolist(), self.image.astype(float),
                      np.zeros((240, 320), dtype=np.uint8),
                      np.zeros((240, 320, 4), dtype=np.uint8),
                      np.zeros((15, 320, 3), dtype=np.uint8)]:
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                DegradeView(image)


if __name__ == "__main__":
    unittest.main()
