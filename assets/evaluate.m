function acc_category_identification = identify_category(predicted_Y)
%% predicted_Y: trials * space dimensions
% global variables "D" for directory path and "P" for subject information.
global D P
​
% "val_index" that specifies the indices of the validation trials: 1*300 (trials)
​
load(fullfile(D.dp.exp_data, 'rand', P.sbj.name{P.ind.sbj}, 'val_index.mat'), 'val_index');
​
% space.vec: 50 (images) *512 (CLIP dimension)
space = load(fullfile(D.dp.exp_data, 'val', 'clip_image.vec.mat'));
n_space_vec = size(space.vec, 1);  % number of image vectors
​
acc_tmp = zeros(size(predicted_Y,1), 1);
% iterating over each predicted vector
for i_pred = 1:size(predicted_Y,1)
    space_corr = zeros(n_space_vec, 1);
    % iterating over each image vector
    % calculating the correlation coefficient between the current predicted
    % vector and the image vector using the "corrcoef" function
    for i_space = 1:n_space_vec
        R = corrcoef(predicted_Y(i_pred,:), space.vec(i_space,:));
        space_corr(i_space) = R(1,2);
    end
    % assigning the index of the current predicted vector to "i_image"
    i_image = val_index(i_pred);
    %  calculating the accuracy of the current predicted vector by counting
    %  the number of image vectors whose correlation coefficients are less
    %  than the correlation coefficient of the corresponding image vector
    %  and dividing by the total number of image vectors minus one
    acc_tmp(i_pred) = numel(find(space_corr<space_corr(i_image)))/(n_space_vec-1);
end
% the overall accuracy of category identification based on all the predicted vectors
acc_category_identification = mean(acc_tmp);
end