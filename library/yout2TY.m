    function [X, TX, Y, TY, R] = yout2TY(yout)
            X = yout.getElement(3).Values.Data;
            TX = yout.getElement(3).Values.Time;
            [TX,ix,~] = unique(TX,'first');
            if size(X,1) == 1 
               X = X(ix);
            else
               X = X(:,ix);
            end
            Y = yout.getElement(2).Values.Data;
            TY = yout.getElement(2).Values.Time;
            [TY,iy,~] = unique(TY,'first');
            if size(Y,1) == 1 
               Y = Y(iy);
            else
               Y = Y(iy,:);
            end
            R = yout.getElement(1).Values.Data;
            R = R(end,1);
    end