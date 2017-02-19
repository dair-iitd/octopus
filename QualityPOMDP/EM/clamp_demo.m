% This script demos how to "clamp" the prior p(Z=1) to ground-truth values
% and shows that doing so can improve accuracy for other inferred labels as well.
%
% It seems that clamping is most useful when good estimates on the prior distributions
% of alpha and beta are not known (see lines 44-45 below).
clear all;

% --------------------------------------------------------------------------
% Initialization
% --------------------------------------------------------------------------
probZ1 = 0.5;
NUM_ITEMS = 100;
NUM_LABELERS = 2;
NUM_RUNS = 50;

for run = 1:NUM_RUNS
	% Initialize labeler and item parameters
	trueAlpha = 4 * rand(NUM_LABELERS, 1);  % alpha in [ 0, 2 ]
	trueBeta = 0.1 + 1 * rand(NUM_ITEMS, 1);  % beta in [ 0.5, 1.5 ]

	% Randomly create ground-truth and given labels
	groundTruth = rand(NUM_ITEMS, 1) < probZ1;  % ground-truth (gold standard) labels

	% Initialize given labels using true parameter values
	idx = 1;
	for i = 1:NUM_LABELERS
		for j = 1:NUM_ITEMS
			itemIds(idx) = j;
			labelerIds(idx) = i;
			if rand < 1/(1+exp(-trueAlpha(i)*trueBeta(j)))  % See our NIPS paper for the model of correctness
				label = groundTruth(j);  % correct
			else
				label = 1 - groundTruth(j);  % incorrect
			end
			labels(idx) = label;
			idx = idx + 1;
		end
	end

	% --------------------------------------------------------------------------
	% Inference
	% --------------------------------------------------------------------------
	% Now, use EM to infer MLE of alpha, beta, and Z.
	priorAlpha = 1;  % Let's give it the wrong prior
	priorBeta = 2;  % Let's give it the wrong prior

	% With clamping:
	% -----------------
	% Suppose the ground-truth labels of the first 1/5th of data are known. Clamp these values by setting p(Z=1) for these items appropriately.
	clampedItems = 1:round(NUM_ITEMS/5);
	otherItems = (max(clampedItems)+1):NUM_ITEMS;
	priorZ1(otherItems) = probZ1;
	priorZ1(clampedItems) = groundTruth(clampedItems) * 0.999 + 0.0001;  % Make probabilities very close, but not quite equal, to 1 or 0.
	[ imageStats, labelerStats ] = em(itemIds, labelerIds, labels, priorZ1, priorAlpha * ones(NUM_LABELERS, 1), priorBeta * ones(NUM_ITEMS, 1));
	accuracyWithClamping(run) = sum((imageStats{2}(otherItems) > 0.5) == groundTruth(otherItems)) / length(otherItems);  % Accuracy just on unclamped data
	%accuracyWithClamping(run) = sum((imageStats{2} > 0.5) == groundTruth) / NUM_ITEMS;  % Accuracy on all data

	% Without clamping:
	% -----------------
	priorZ1(1:NUM_ITEMS) = probZ1;
	[ imageStats, labelerStats ] = em(itemIds, labelerIds, labels, priorZ1, priorAlpha * ones(NUM_LABELERS, 1), priorBeta * ones(NUM_ITEMS, 1));
	accuracyWithoutClamping(run) = sum((imageStats{2}(otherItems) > 0.5) == groundTruth(otherItems)) / length(otherItems);  % Accuracy just on unclamped data
	%accuracyWithoutClamping(run) = sum((imageStats{2} > 0.5) == groundTruth) / NUM_ITEMS;  % Accuracy on all data
end

disp(sprintf('Mean accuracy with clamping: %f    Mean accuracy without clamping: %f', mean(accuracyWithClamping), mean(accuracyWithoutClamping)));
