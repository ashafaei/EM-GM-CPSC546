%% Alireza Shafaei
% This code generates random data and runs em_gm.
% Since the initialization is random, make sure you run this multiple times
% to get satisfactory results.

%% Generate data

MU1 = [40 40];
SIGMA1 = 80* [2 0;
              0 0.2];
MU2 = [50 60];
SIGMA2 = 40* [1 0;
              0 1];
MU3    = [20 65];
SIGMA3 = 25*[0.2 0;
              0   2];
X1 = [mvnrnd(MU1,SIGMA1,200)];
X2 = [mvnrnd(MU2,SIGMA2,200)];
X3 = [mvnrnd(MU3,SIGMA3,300)];

subplot(1, 2, 1);
scatter([X1(:,1)],[X1(:, 2)],10,'ro');
hold on;
scatter([X2(:, 1)],[X2(:,2)],10,'bo');
scatter([X3(:, 1)],[X3(:,2)],10,'go');
hold off;

xlabel('X');
ylabel('Y');
title('Point set X');
axis equal;

X = [X1; X2; X3];

subplot(1, 2, 2);
scatter(X(:, 1), X(:, 2),10,'bo');
xlabel('X');
ylabel('Y');
title('Point set X');
axis equal;

%% Solve the clustering

clusters = 3;
while true
    try
        [Prior, Sigmas, Mus] = EM_gm(X, clusters);
        break;
    catch exception
    end
end

%% Plot the output
for j=1:clusters,
    x = 0:1:100;
    y = 0:1:100;
    [X Y] = meshgrid(x,y);

    Z = mvnpdf([X(:) Y(:)], Mus(:, j)',diag(Sigmas(:, j).^2));
    Z = reshape(Z,size(X));
    hold on;
    contour(X,Y,Z), axis equal
    hold off;
end