function [data,a,b] = createScaledData(matrix,low,high,lowerBuffer,upperBuffer)
%  Takes a matrix of values and scales them to between the high and low values
%  Scale for each row!
%    the upper and lower buffers can be defined to create more robust scaling factors
%    data is the returned conditioned (already scaled) array
%    a is the multiplying factor
%    b is the additional bias value
%    f(x) = ax + b  creates the scaled value

% TO-DO: NEED To Deal with -901 issues

%   | a1 b1 c1 |     | a1+b1+c1 |
%   | a2 b2 c2 |  =  | a2+b2+c2 |
%   | a3 b3 c3 |     | a3+b3+c3 |

if (nargin < 5)
    upperBuffer = 0.0;
end
if (nargin < 4)
    lowerBuffer = 0.0;
end

arr = matrix;

maxVal = max(arr,[],2)+upperBuffer;
minVal = min(arr,[],2)-lowerBuffer;
a = (high-low)./max(maxVal-minVal,1e-5);
b = low - (a.*minVal);

data = arr.*a + b;
