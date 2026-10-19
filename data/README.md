# IDRiD Dataset Reference

## Overview
The Indian Diabetic Retinopathy Image Dataset (IDRiD) is an openly accessible dataset of retinal fundus images acquired from an eye clinic located in Nanded, Maharashtra, India.

- **Citation**: Porwal, P., Pachade, S., Kamble, R., Kokare, M., Deshmukh, G., Sahasrabuddhe, V., & Meriaudeau, F. (2018). *Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research*. Data, 3(3), 25.
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0).

## Workspace Directory Structure
The dataset in this workspace is organized into:
- `B. Disease Grading/`
  - `1. Original Images/`
    - `a. Training Set/` (413 fundus images: `IDRiD_001.jpg` to `IDRiD_413.jpg`)
    - `b. Testing Set/` (103 fundus images: `IDRiD_001.jpg` to `IDRiD_103.jpg`)
  - `2. Groundtruths/`
    - `a. IDRiD_Disease Grading_Training Labels.csv` (413 samples)
    - `b. IDRiD_Disease Grading_Testing Labels.csv` (103 samples)
- `A. Segmentation/` (Pixel-level lesion masks: Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates, Optic Disc)
- `C. Localization/` (Centre of Optic Disc and Fovea coordinates)

For the DR severity classification project, the pipeline operates on `B. Disease Grading`.

## Image Specifications
- **Image Format**: RGB JPEG
- **Native Resolution**: 4288 × 2848 pixels
- **Field of View (FOV)**: 50 degrees
- **Camera**: Kowa VX-10 alpha digital fundus camera

## Diabetic Retinopathy Severity Grading Scale (ICDR Standard)
- **Grade 0**: No Diabetic Retinopathy (No abnormalities)
- **Grade 1**: Mild Non-Proliferative DR (Microaneurysms only)
- **Grade 2**: Moderate Non-Proliferative DR (More than microaneurysms, but less than severe NPDR)
- **Grade 3**: Severe Non-Proliferative DR (4-2-1 rule: severe intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant)
- **Grade 4**: Proliferative DR (Neovascularization, vitreous/preretinal hemorrhage)
