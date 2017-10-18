function image = showLandmarks(landmarks, image)
    if(size(landmarks,1)~=2 && size(image,3)~=3)
        return;
    end

    for i=1:size(landmarks,2)
        landmarks(2,i) = size(image,1)-landmarks(2,i);
    end

    landmarks = int32(landmarks);

    for i=1:3
        landmarks(3,:) = i;
        idx = sub2ind(size(image),landmarks(2,:),landmarks(1,:),landmarks(3,:));
        if(i==2)
            image(idx) = 255;
        else
            image(idx) = 0;
        end
    end
end