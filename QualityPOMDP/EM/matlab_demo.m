% matlab_demo.m -- demonstrates how the inference algorithm can more accurately deduce ground-truth labels
% than majority vote heuristic. This is a simulation. Expected output:
%
%  ...
% Accuracy using optimal inference: 0.999000
% Accuracy using majority vote: 0.942000
% Mean inferred alpha for regular labelers: 0.356239
% Mean inferred alpha for expert labelers: 2.115111
% Mean inferred beta for easy images: 3.833752
% Mean inferred beta for hard images: 1.522822
%
% Note that this script also writes a file called "data.txt" which may be passed to the C command-line
% interface as:
%  ./em data.txt

% Simulation parameters
NUM_REGULAR_LABELERS = 16;
NUM_EXPERT_LABELERS = 4;
NUM_HARD_IMAGES = 300;
NUM_EASY_IMAGES = 700;
NUM_LABELERS = NUM_REGULAR_LABELERS + NUM_EXPERT_LABELERS;
NUM_IMAGES = NUM_HARD_IMAGES + NUM_EASY_IMAGES;

P_CORRECT_EASY_REGULAR = 0.8;
P_CORRECT_EASY_EXPERT = 0.99;
P_CORRECT_HARD_REGULAR = 0.5;
P_CORRECT_HARD_EXPERT = 0.95;

P_Z1 = 0.5;
rand('seed', 0);

% Sample ground-truth labels
for i = 1:NUM_IMAGES
	trueLabels(i,1) = rand() <= P_Z1;
end

% Sample the observed labels
givenLabels = zeros(NUM_IMAGES * NUM_LABELERS, 1);
imageIds = zeros(NUM_IMAGES * NUM_LABELERS, 1);
labelerIds = zeros(NUM_IMAGES * NUM_LABELERS, 1);
idx = 1;
fp = fopen('data.txt', 'wt');
fprintf(fp, '%d %d %d %f\n', NUM_IMAGES * NUM_LABELERS, NUM_LABELERS, NUM_IMAGES, P_Z1);
for i = 1:NUM_IMAGES
	for j = 1:NUM_LABELERS
		% Determine probability of correct label given image and labeler types
		if i > NUM_EASY_IMAGES
			if j > NUM_REGULAR_LABELERS
				p = P_CORRECT_HARD_EXPERT;
			else
				p = P_CORRECT_HARD_REGULAR;
			end
		else
			if j > NUM_REGULAR_LABELERS
				p = P_CORRECT_EASY_EXPERT;
			else
				p = P_CORRECT_EASY_REGULAR;
			end
		end	
		
		% If correct, then given label is ground-truth; otherwise, it's the opposite
		correct = rand() <= p;
		if correct
			givenLabel = trueLabels(i);
		else
			givenLabel = 1 - trueLabels(i);
		end
		fprintf(fp, '%d %d %d\n', i - 1, j - 1, givenLabel);  % "- 1" -- IDs must start at 0 for C interface!!

		% Create the parallel arrays of image IDs, labeler IDs, and given labels
		imageIds(idx) = i;
		labelerIds(idx) = j;
		givenLabels(idx) = givenLabel;

		idx = idx + 1;
	end
end
fclose(fp);

% Perform EM to infer pZ, beta, and alpha
[ imageStats, labelerStats ] = em(imageIds, labelerIds, givenLabels, P_Z1, ones(NUM_LABELERS, 1), ones(NUM_IMAGES, 1));

% Infer image labels by thresholding pZ; report accuracy
inferredLabels = imageStats{2} >= 0.5;
disp(sprintf('Accuracy using optimal inference: %f', sum(inferredLabels == trueLabels) / NUM_IMAGES));

% Infer image labels by majority vote; report accuracy
for i = 1:NUM_IMAGES
	idxs = find(imageIds == i);
	majorityVoteLabels(i,1) = round(sum(givenLabels(idxs)) / length(idxs));
end
disp(sprintf('Accuracy using majority vote: %f', sum(majorityVoteLabels == trueLabels) / NUM_IMAGES));

% Display info about other stats
disp(sprintf('Mean inferred alpha for regular labelers: %f', mean(labelerStats{2}(1:NUM_REGULAR_LABELERS))));
disp(sprintf('Mean inferred alpha for expert labelers: %f', mean(labelerStats{2}(NUM_REGULAR_LABELERS+1:end))));
disp(sprintf('Mean inferred beta for easy images: %f', mean(imageStats{3}(1:NUM_EASY_IMAGES))));
disp(sprintf('Mean inferred beta for hard images: %f', mean(imageStats{3}(NUM_EASY_IMAGES+1:end))));
