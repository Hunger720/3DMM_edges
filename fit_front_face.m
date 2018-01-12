% clear all
% 
% addpath('ZhuRamananDetector','optimisations','utils');
% 
% % YOU MUST set this to the base directory of the Basel Face Model
% load('01_MorphableModel.mat');
% % Important to use double precision for use in optimisers later
% shapeEV = double(shapeEV);
% shapePC = double(shapePC);
% shapeMU = double(shapeMU);
% 
% % We need edge-vertex and edge-face lists to speed up finding occluding boundaries
% % Either: 1. Load precomputed edge structure for Basel Face Model
% load('BFMedgestruct.mat');
% % Or:     2. Compute lists for new model:
% % TR = triangulation(tl,ones(k,1),ones(k,1),ones(k,1));
% % Ev = TR.edges;
% % clear TR;
% % Ef = meshFaceEdges( tl,Ev );
% % save('edgestruct.mat','Ev','Ef');
% 

%% Adjustable Parameters
% 
% % Number of model dimensions to use: max 199
% ndims = 199;
% % Prior weight for initial landmark fitting
% w_initialprior = 0.7;
% % Number of iterations for iterative closest edge fitting
% icefniter = 7;
% 
% options.Ef = Ef;
% options.Ev = Ev;
% 
% options.w1 = 0.45; % w1 = weight for edges
% options.w2 = 0.15; % w2 = weight for landmarks
%                    % w3 = 1-w1-w2 = prior weight
%

%% Setup Basic Parameters
% 
% imPath = 'testImages/171122/front_320.png';
% im = imread(imPath);
% edgeim = edge(rgb2gray(im),'canny',0.15);
% x_landmarks = loadLandmarks(im, 'testImages/171122/front_320_86landmarks.txt'); 
% landmarks = loadCorrespondence('correspondence/correspondence86.txt'); 
% 
% %% Initialise using only landmarks (unnecessary because it is also done is FitEdges)
% 
% % disp('Fitting to landmarks only...');
% % [b,R,t,s] = FitSingleSOP( x_landmarks,shapePC,shapeMU,shapeEV,ndims,landmarks,w_initialprior );
% % FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
% FV.faces = tl;
% 

%% Initialise Using Iterative Closest Edge Fitting (ICEF)

% disp('Fitting to edges with iterative closest edge fitting...');
% [b,R,t,s]   = FitEdges(im,x_landmarks,landmarks,shapePC,shapeMU,shapeEV,options.Ef,options.Ev,tl,ndims, w_initialprior, options.w1, options.w2,icefniter);
% FV.vertices = reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
% FV.faces    = tl;
% 
% % Run final optimisation of hard edge cost
% 
% disp('optimising non-convex edge cost...');
% maxiter = 5;
% iter    = 0;
% diff    = 1;
% eps     = 1e-9;
% 
% [r,c] = find(edgeim);
% r = size(edgeim,1)+1-r;
% 
% while (iter<maxiter) && (diff>eps)
%     
%     FV.vertices = reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
%     [ options.occludingVertices ] = occludingBoundaryVertices( FV,options.Ef,options.Ev,R );
% 
%     X = reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC(:,1:ndims),1)/3);   
%     % Compute position of projected occluding boundary vertices
%     x_edge = R*X(:,options.occludingVertices);
%     x_edge = x_edge(1:2,:);
%     x_edge(1,:)=s.*(x_edge(1,:)+t(1));
%     x_edge(2,:)=s.*(x_edge(2,:)+t(2));
%     % Find edge correspondences
%     [idx,d] = knnsearch([c r],x_edge');
%     % Filter edge matches - ignore the worse 5% 
%     sortedd=sort(d);
%     threshold = sortedd(round(0.95*length(sortedd)));
%     idx = idx(d<threshold);
%     options.occludingVertices = options.occludingVertices(d<threshold);
% 
%     b0 = b;
%     [ R,t,s,b ] = optimiseHardEdgeCost( b0,x_landmarks,shapeEV,shapeMU,shapePC,R,t,s,r,c,landmarks,options,tl,false );
%     
%     diff = norm(b0-b);
%     disp(num2str(diff));
%     iter = iter+1;
%     
% end
% 
% % Run optimisation for a final time but without limit on number of iterations
% [ R,t,s,b ] = optimiseHardEdgeCost( b,x_landmarks,shapeEV,shapeMU,shapePC,R,t,s,r,c,landmarks,options,tl,true );


%% Segmented Optimise

FV.vertices = reshape(shapePC(:,1:ndims)*b+shapeMU, 3, size(shapePC,1)/3)';

landmarksIdx = landmarks;

noseLandmarksOrd = [28:31 64:80];
noseLandmarksIdx = landmarksIdx(noseLandmarksOrd);
noseLandmarks    = x_landmarks(:, noseLandmarksOrd);
% eyesLandmarksOrd  = [18:27 32:43 81:86];
% eyesLandmarksIdx  = landmarksIdx(eyesLandmarksOrd);
% eyesLandmarks     = x_landmarks(:, eyesLandmarksOrd);
% mouthLandmarksOrd = 44:63;
% mouthLandmarksIdx = landmarksIdx(mouthLandmarksOrd);
% mouthLandmarks    = x_landmarks(:, mouthLandmarksOrd);
% restLandmarksOrd  = 1:17;
% restLandmarksIdx  = landmarksIdx(restLandmarksOrd);
% restLandmarks     = x_landmarks(:, restLandmarksOrd);

optoptions = optimoptions('lsqnonlin','Display','iter','Algorithm','levenberg-marquardt','TolX',1e-10,'TolFun',1e-10);
wLandmarks = 10;
wPrior     = 1; 

disp('optimising nose ...');
noseCoef  = lsqnonlin(@(b) segmentedOptimiseCost(b, R, s, t, shapeMU, shapePC, shapeEV, noseLandmarksIdx, noseLandmarks, wLandmarks, wPrior), b, [], [], optoptions);
% disp('optimising eyes ...');
% eyesCoef  = lsqnonlin(@(b) segmentedOptimiseCost( b, R, s, t, shapeMU, shapePC, shapeEV, eyesLandmarksIdx, eyesLandmarks, wLandmarks, wPrior ), b, [], [], optoptions);
% disp('optimising mouth ...');
% mouthCoef = lsqnonlin(@(b) segmentedOptimiseCost( b, R, s, t, shapeMU, shapePC, shapeEV, mouthLandmarksIdx, mouthLandmarks, wLandmarks, wPrior ), b, [], [], optoptions);
% disp('optimising rest ...');
% restCoef  = lsqnonlin(@(b) segmentedOptimiseCost( b, R, s, t, shapeMU, shapePC, shapeEV, restLandmarksIdx, restLandmarks, wLandmarks, wPrior ), b, [], [], optoptions);

% % recalculate the segmented coefficients or remain the global optimise coefficient value
% noseCoef  = b;
eyesCoef  = b;
mouthCoef = b;
restCoef  = b;
shapeCoef = [noseCoef eyesCoef mouthCoef restCoef];

shape = coef2object(shapeCoef, shapeMU, shapePC, ones(size(shapeEV)), segMM, segMB);
FV.vertices = reshape(shape, 3, size(shape,1)/3)';


%% Texture Sampling

% disp('texture sampling ...');
% 
% FV.facevertexcdata = reshape(texPC(:,1:ndims)*ones(ndims,1)+texMU,3,size(texPC,1)/3)';
% FV = textureSampling(FV, im, R, t, s);


%% Optimise Lighting

% disp('optimising lighting ...');
% 
% % initial render parameters
% renderParams      = ones(9,1);
% renderParams(1:3) = [1 1 1]';
% renderParams(4:6) = [1 1 1]';
% renderParams(7:9) = [0 0 1]';
% 
% renderParams  = lsqnonlin(@(renderParams) lightOptimiseCost(renderParams, FV, R, s, t, im), renderParams);
% 
% oglp.height      = size(im,1);
% oglp.width       = size(im,2);
% oglp.i_amb_light = 1.0 .* renderParams(1:3)';
% oglp.i_dir_light = 1.0 .* renderParams(4:6)';
% oglp.d_dir_light = renderParams(7:9);
% oglp.shininess   = 1;
% oglp.specularity = 0;


%% Optimise Texture

% disp( 'optimising texture ...' );
% 
% wPrior  = 0.1;
% texCoef = ones(size(texEV));
% texCoef = lsqnonlin(@(texCoef) textureOptimiseCost(texCoef, R, s, t, oglp, FV, im, texMU, texPC, texEV, wPrior), texCoef);
% 
% FV.facevertexcdata = reshape(texPC(:,1:ndims)*texCoef+texMU, 3, size(texPC,1)/3)';
% FV = textureSampling(FV, im, R, t, s);


%% UV Texture Mapping

% load('BFM_UV.mat');
% fileName = 'uvTexture/mix_tex_image.png';
% texImage = imread(fileName);
% FV = uvTextureMapping(FV, UV, texImage);


%% Results

disp('rendering final results...');

% showLandmarks(x_landmarks, im, true);

[image, xx_landmarks] = showLandmarksOnModel(FV, R, s, t, oglp, im, landmarks);
figure; imshow(image);

% figure; distance(x_landmarks, xx_landmarks)

% figure; plot(x_landmarks(1,:), x_landmarks(2,:), 'go', xx_landmarks(1,:), xx_landmarks(2,:), 'ro');


%% Save As ply File

disp('saving as ply file ...');

tex   = FV.facevertexcdata';
tex   = tex(:);

% % show landmark vertices on the model
% for i = 1:size(landmarks,1)
%     tex( 1 + 3*(landmarks(i)) ) = 0;
%     tex( 2 + 3*(landmarks(i)) ) = 255;
%     tex( 3 + 3*(landmarks(i)) ) = 0;
% end

plywrite('ply/180111/front_face.ply', shape, tex, tl );