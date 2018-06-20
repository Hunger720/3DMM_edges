%% Merging Landmarks

landmarksPath  = 'testImages/180618/face_98landmarks.txt';
landmarksPath1 = 'testImages/180618/face_68landmarks.txt';
landmarksPath2 = 'testImages/180618/face_194landmarks.txt';
landmarksPath3 = 'testImages/180618/ear_landmark.txt';
index1 = [1:31 37:68];
index2 = [42:58 188 190 192 172 170 168];
landmarks1 = load(landmarksPath1);
landmarks2 = load(landmarksPath2);
landmarks3 = load(landmarksPath3);
landmarks = [landmarks1(index1,:); landmarks2(index2,:); landmarks3];

fid = fopen(landmarksPath, 'w');
[r c] = size(landmarks);
for i = 1:r
    for j = 1:c
        if j == c
            fprintf(fid, '%g\n', landmarks(i,j));
        else
            fprintf(fid, '%g\t', landmarks(i,j));
        end
    end
end
fclose(fid);

% show merging result
im = imread('testImages/180618/face.png');
landmarks = loadLandmarks(im, landmarksPath); 
showLandmarks(landmarks, im, true);