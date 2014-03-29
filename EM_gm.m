function [ Prior, Sigmas, Mus ] = EM_gm( datapoints, clusters, max_iter )
%EM_GM Solves the isotropic gaussian mixture model problem.
%   Inputs:
%       datapoints: A mxn matrix of m samples with n dimensions.
%       clusters:   The expected number of clusters
%       max_iter:   The maximum number of iterations (default = 100).
%   Outputs:
%       Prior:  The prior \Pi
%       Sigmas: The Sigmas for each Gaussian Model
%       Mus:    The means for each Gaussian Model.

if (nargin < 3)
    max_iter = 100;
end

% Determine what's the dimension of datapoints (eg R^n, n=?)
space = size(datapoints, 2);

% The Sigma of gaussians.
Sigmas = ones([space clusters])*10;
Mus    = datasample(datapoints, clusters)';
Prior  = randn(clusters, 1);
% Normalize the prior.
Prior  = Prior/sum(Prior);

% determine number of datapoints.
m = size(datapoints, 1);

for iteration = 1:max_iter,
    % Calculating the log likelihood (\lambda)
    gauss_vals  = zeros(m, clusters);
    for j=1:clusters,
       gauss_vals(:, j) = mvnpdf(datapoints, Mus(:, j)', diag(Sigmas(:, j).^2));
    end
    lambda = sum(log(sum(gauss_vals, 2)));
    fprintf('%d: Log Likelihood: %.2f\n', iteration, lambda);
    
    % The E step P(k|x)
    p_k_given_x = zeros(m, clusters);
    
    gauss_vals  = zeros(m, clusters);
    for j=1:clusters,
       gauss_vals(:, j) = mvnpdf(datapoints, Mus(:, j)', diag(Sigmas(:, j).^2));
    end
    
    for j=1:clusters,
       p_k_given_x(:, j) = (Prior(j).*gauss_vals(:, j)) ./ (gauss_vals * Prior);
    end
    % Now we have calculated the formula 5 from the review.
    
    % The M Step, calculating 20, 21, and 22 from the review.
    new_Mus = Mus;
    
    for j=1:clusters,
        new_Mus(:, j) = (p_k_given_x(:, j)'*datapoints ./ sum(p_k_given_x(:, j)))';
        Sigmas(:, j) = ...
            sqrt(1/space * ...
                    (p_k_given_x(:, j)'* (datapoints-repmat(Mus(:, j)', m, 1)).^2)...
                    / sum(p_k_given_x(:, j)));
        Prior(j) = 1/m * sum(p_k_given_x(:, j));
    end
    
    Mus = new_Mus;
end

end