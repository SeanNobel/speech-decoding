EEG Datasets for Naturalistic Listening to "Alice in Wonderland"
Jonathan Brennan <jobrenn@umich.edu>
ORCID: 0000-0002-3639-350X
https://sites.lsa.umich.edu/cnllab


Description:
These files contain the raw data and processing parameters to go with the paper "Hierarchical structure guides rapid linguistic predictions during naturalistic listening" by Jonathan R. Brennan and John T. Hale. These files include the stimulus (wav files), raw data (matlab format for the Fieldtrip toolbox), data processing paramaters (matlab), and variables used to align the stimuli with the EEG data and for the statistical analyses reported in the paper.


Method:
The data comprise 49 human electroencephalography (EEG) datasets collected at the University of Michigan Computational Neurolinguistics Lab. The data were recorded with 61 active electrodes and a Brain Products actiCHamp amplifier at 500 Hz (0.1 to 200 hz band). Participants listened passively to a 12.4 m audiobook recording of the first chapter of Alice's Adventures in Wonderland (librivox.org, catalog date 2006-01-12) and after which they completed a short 8-question comprehension questionnaire. The raw data are stored as MATLAB (version 2016a) data structures created by the Fieldtrip toolbox (version 20170322, available at http://fieldtriptoolbox.org/) 

Data Set Description:

"audio.zip" contains 12 stimulus files
  - chapter one of "Alice's Adventures in Wonderland" from librivox.org
  - divided into 12 .wav files

"S01.mat" through "S49.mat" 49 EEG datasets
  - Matlab structures converted for use with the Fieldtrip Toolbox

"proc.zip" contains pre-PROC-essing parameters for 42 datasets
  - Matlab
  - 7 datasets not represented as these were too noisy to pre-process
  - includes channel rejections, epoch rejections, ICA unmixing matrix etc.

"datasets.mat"
  - matlab file with variables indicating which datasets were:
    - N=33 that were USEd in the main analysis
    - N=8 that were excluded due to LOW PERFormance on the comprehension quiz
    - N=8 that come from participants with HIGH NOISE (N=8)
    - (note that 2 participants had both high noise and low performance!)

"comprehension-questions.doc
  - Multiple choice comprehension questions

"comprehension-scores.txt"
  - Score out of 8 on a post-experiment comprehension questionnaire
  - Also includes dataset rejection comments (due to behavioral results and/or noise)
  - errata:
    - Three participants skipped 4 questions; they received scores out of 4
    - S21's score could not be found when compiling this table
    - S39 was included in the original analysis despite not meeting behavioral criteria. (Note that the paper includes analysis disregarding behavioral criteria showing that this does not impact the results)

"AliceChapterOne-EEG.csv"
  - csv file with word-by-word variables used for data analysis
  - includes NGRAM, RNN, and CFG surprisal
  - The spreadsheet has 16 columns and 2130 rows
  - Each row is a word in the first chapter of Alice's Adventures in Wonderland
  - Col 1 "Word" word token 
  - Col 2 "Segment" 1 through 12 indicating which of twelve audio segments this word appeared in
  - Col 3 "onset" word's onset time in seconds relative to the beginning of the current segment
  - Col 4 "offset" word's offset time in sec relative to the beginning of the current segment
  - Col 5 "Order" Indicates word order (1, 2, 3... 2129) within full stimulus
  - Col 6 "LogFreq" log-transformed word frequency from the English Lexicon Project HAL corpus
  - Col 7, 8 "LogFreq_Prev" "LogFreq_Next" same measure for the previous and next word
  - Col 9 "SndPower" Average power of auditory stimulus over 50ms after word onset
  - Col 10 "Length" Word length in sec (offset - onset)
  - Col 11 "Position" Indicates word position (1, 2, 3...) within each sentence
  - Col 12 "Sentence" Indicates sentence position (1, 2, 3... 84) within full stimulus
  - Col 13 "IsLexical" Logical indicating whether the word is a lexical/content (1) or function word (0)
  - Col 14 "NGRAM" Surprisal values from the NGRAM language model
  - Col 15 "RNN" Surprisal values from the RNN language model
  - Col 16 "CFG" Surprisal values from the CFG language model 

This data set can be cited as:

Brennan, J.R. (2018). EEG Datasets for Naturalistic Listening to ""Alice in Wonderland"" [Data set]. University of Michigan Deep Blue Data Repository. 
https://doi.org/10.7302/Z29C6VNH


References:

Brennan, J. R., & Hale, J. T. (To appear). Hierarchical structure guides rapid linguistic predictions during naturalistic listening. PLoS ONE

Robert Oostenveld, Pascal Fries, Eric Maris, and Jan-Mathijs Schoffelen. (2011) FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive Electrophysiological Data. Computational Intelligence and Neuroscience, vol. 2011. doi:10.1155/2011/156869.


History:

- December 10 2018: Initial Release
