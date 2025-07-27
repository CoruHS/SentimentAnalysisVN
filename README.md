# VietAICode
Check this out bob

OK BUT PLEASE READ THIS.

Anways heres a brief description/time line:

Timeline of architectures(essentially which architectures i chose)
NLPS --> NLPS + data augmentation/other shenanigans --> CNN + NLPS

I start the script by loading three NumPy arrays: two that hold the labeled training data and one that holds the public test set. As soon as those arrays are in memory, I scan the training sentences to figure out the largest token ID; that single check tells me exactly how big the embedding table has to be. Every sentence then flows into a custom Dataset that can roughen the input on the fly: sometimes I drop individual tokens at random, and other times I blank out a short span by replacing its tokens with zeros. This little bit of noise forces the network to rely on the remaining context instead of memorizing any one phrasing.

For the model itself I keep things compact but effective. I feed token IDs through an embedding layer that turns them into 256‑dimensional vectors. Next, four separate one‑dimensional convolutional filters—sizes two through five—scan over the sentence. After each filter runs, I take a max over time to capture the most salient feature it found. I then concatenate those four pooled vectors, apply dropout to discourage overfitting, and send the result through a fully connected layer that produces three raw scores, one for each sentiment class: negative, neutral, and positive. The architecture is classic TextCNN, but it stays light enough to train quickly on an ordinary laptop GPU.

Where I really depart from a garden‑variety trainer is in the loss function and how I sample data. Instead of plain cross‑entropy, I mix focal loss with label smoothing. The label smoothing gently nudges each one‑hot target toward a uniform distribution, which stops the network from becoming overconfident. Focal loss, on the other hand, down‑weights easy examples so the gradient zeroes in on the hard ones. To keep class imbalance from skewing the training batches, I compute inverse‑frequency weights per class and give an extra nudge to the under‑represented neutral class. That weighting drives the data loader so each mini‑batch carries a fairer mix of sentiments, and the loss function refuses to ignore those tricky neutral examples.

For each random seed I run training in two phases. First I carve out ten percent of the data as a validation split, train for the chosen number of epochs, and print both a confusion matrix and a macro‑averaged F1 score so I can see exactly where the model stumbles. After that sanity check, I reinitialize the model and retrain on the full dataset so every example contributes to the final ensemble. I then run the public test sentences through this freshly trained network, collect their soft‑max outputs, and save them for later. I repeat that entire cycle for however many seeds I requested. In the end I average all those soft‑max matrices and take the arg‑max to get a single prediction for each sentence. That ensemble step usually steadies the results and bumps my score a little higher than any single run could.

I also cleaned up the file itself so it is easier on the eyes. I added Python type hints everywhere, switched to pathlib.Path for safer file handling, scaled CPU thread counts automatically, renamed everything in snake‑case for PEP‑8 compliance, and replaced stiff headers with conversational comments so the flow feels like I am guiding a friend through the code rather than dumping boilerplate. Functionally the script is still the same TextCNN pipeline, but with modern regularization tricks, an ensemble wrapper, and a friendlier voice, I can both understand it quickly and trust it to squeeze a few extra points out of my sentiment‑analysis task.
