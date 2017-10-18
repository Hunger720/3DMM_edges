function [x_landmarks, landmarks] = loadLandmarks(image, pointsFile, indexFile)
    x_landmarks = load(pointsFile);
    x_landmarks = x_landmarks';
    for i=1:size(x_landmarks,2)
        x_landmarks(2,i) = size(image,1)-x_landmarks(2,i);
    end
    
    landmarks = load(indexFile);
    landmarks = landmarks(:, 2);
end