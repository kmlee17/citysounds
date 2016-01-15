### CitySounds Presentation
* Introduction
* Play short audio clips
	- Just from those short sounds, we can identify what that sound is.  My question is how good is a machine in doing this same task.
* Dataset explanation
	- 10 classes: Air Conditioner, Car Horn, Children Playing, Dog Bark, Drilling, Engine Idling, Gun Shot, Jackhammer, Siren, Street Music
	- 8732 samples (400-1000 per class, no major class imbalance)
* Data pipeline
	- Raw audio
	- Loaded in as a matrix
	- Transformed in the frequency domain using Fourier Transforms
	- Run through Mel Frequency filters and a spectrogram with specified parameters to extract Mel Frequency Cepstral Coefficients (MFCCs)
	- Mean, variance, first and second derivative, min, max, skew, kurtosis of MFCCs are taken to create a feature vector of each audio sample
	- LDA
	- SVM
* Percentage confusion matrix of results
	- The colored diagonal represents how often the sound was correctly predicted
	- Highlight a few of the more prominant missclassifications
		* Jackhammer vs drill
		* Ambient noises (air conditioner vs engine idling)
		* Model is definitely better at picking up more impactful sounds like gun shots or police sirens
* Web App
	- Users can upload a short audio wav file from the 10 classes and the app will predict what that sound is.
* Future work
   - Explore techniques to better extract features from ambient sounds
   - Apply to different sound classifications (tap dance sounds, etc)