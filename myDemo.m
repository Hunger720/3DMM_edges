clear all

%% 
addpath('ZhuRamananDetector','optimisations','utils','comparison');

% YOU MUST set this to the base directory of the Basel Face Model
BFMbasedir = '';

% Load morphable model
load(strcat(BFMbasedir,'01_MorphableModel.mat'));
% Important to use double precision for use in optimisers later
shapeEV = double(shapeEV);
shapePC = double(shapePC);
shapeMU = double(shapeMU);

% We need edge-vertex and edge-face lists to speed up finding occluding boundaries
% Either: 1. Load precomputed edge structure for Basel Face Model
load('BFMedgestruct.mat');
% Or: 2. Compute lists for new model:
%TR = triangulation(tl,ones(k,1),ones(k,1),ones(k,1));
%Ev = TR.edges;
%clear TR;
%Ef = meshFaceEdges( tl,Ev );
%save('edgestruct.mat','Ev','Ef');

%% ADJUSTABLE PARAMETERS

% Number of model dimensions to use: max 199
ndims = 199;
% Prior weight for initial landmark fitting
w_initialprior = 1.0;
% Number of iterations for iterative closest edge fitting
icefniter=7;

options.Ef = Ef;
options.Ev = Ev;
% w1 = weight for edges
% w2 = weight for landmarks
% w3 = 1-w1-w2 = prior weight
options.w1 = 0.3; 
options.w2 = 0.3;

%% Setup basic parameters

testdir='testImages/';
im = imread(strcat(testdir,'front.bmp'));
edgeim = edge(rgb2gray(im),'canny',0.15);
[x_landmarks, landmarks] = loadLandmarks(im, 'landmarks80.txt', 'correspondence80.txt');

%% Initialise using only landmarks

disp('Fitting to landmarks only...');
[b,R,t,s] = FitSingleSOP( x_landmarks,shapePC,shapeMU,shapeEV,ndims,landmarks,w_initialprior );
FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
FV.faces = tl;

%% Initialise using iterative closest edge fitting (ICEF)

disp('Fitting to edges with iterative closest edge fitting...');
[b,R,t,s] = FitEdges(im,x_landmarks,landmarks,shapePC,shapeMU,shapeEV,options.Ef,options.Ev,tl,ndims, w_initialprior, options.w1, options.w2,icefniter);
FV.vertices = reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';

%% Run final optimisation of hard edge cost

disp('Optimising non-convex edge cost...');
maxiter = 5;
iter = 0;
diff = 1;
eps = 1e-9;

[r,c]=find(edgeim);
r = size(edgeim,1)+1-r;

while (iter<maxiter) && (diff>eps)
    
    FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
    [ options.occludingVertices ] = occludingBoundaryVertices( FV,options.Ef,options.Ev,R );

    X = reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC(:,1:ndims),1)/3);   
    % Compute position of projected occluding boundary vertices
    x_edge = R*X(:,options.occludingVertices);
    x_edge = x_edge(1:2,:);
    x_edge(1,:)=s.*(x_edge(1,:)+t(1));
    x_edge(2,:)=s.*(x_edge(2,:)+t(2));
    % Find edge correspondences
    [idx,d] = knnsearch([c r],x_edge');
    % Filter edge matches - ignore the worse 5% 
    sortedd=sort(d);
    threshold = sortedd(round(0.95*length(sortedd)));
    idx = idx(d<threshold);
    options.occludingVertices = options.occludingVertices(d<threshold);

    b0 = b;
    [ R,t,s,b ] = optimiseHardEdgeCost( b0,x_landmarks,shapeEV,shapeMU,shapePC,R,t,s,r,c,landmarks,options,tl,false );
    
    diff = norm(b0-b);
    disp(num2str(diff));
    iter = iter+1;
    
end

% Run optimisation for a final time but without limit on number of
% iterations
[ R,t,s,b ] = optimiseHardEdgeCost( b,x_landmarks,shapeEV,shapeMU,shapePC,R,t,s,r,c,landmarks,options,tl,true );

%% Results
disp('Rendering final results...');

FV.vertices=reshape(shapePC(:,1:ndims)* b +shapeMU,3,size(shapePC,1)/3)';
FV.facevertexcdata = reshape(texPC(:,1:ndims)*ones(ndims,1)+texMU,3,size(texPC,1)/3)';

showLandmarks(x_landmarks, im, true);

[image, xx_landmarks] = showLandmarksOnModel(FV, R, s, t, im, landmarks);
figure; imshow(image);

figure; distance(x_landmarks, xx_landmarks)

figure; plot(x_landmarks(1,:), x_landmarks(2,:), 'go', xx_landmarks(1,:), xx_landmarks(2,:), 'ro');

%% Iterarive Detail Refinement

%% Save as ply file
% shape = coef2object(b, shapeMU, shapePC(:,1:ndims), ones(ndims,1));
% tex   = coef2object(ones(ndims,1), texMU, texPC(:,1:ndims), ones(ndims,1));

% % show landmark vertices on the model
% for i = 1:size(landmarksx,1)
%     tex( 1 + 3*(landmarks(i)) ) = 0;
%     tex( 2 + 3*(landmarks(i)) ) = 255;
%     tex( 3 + 3*(landmarks(i)) ) = 0;
% end

% plywrite('fit_edges_redefined_landmarks_head.ply', shape, tex, tl );
% clear shape tex;