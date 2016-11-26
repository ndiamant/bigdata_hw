function X = Final1(c)
    a = dir('DAILY');
    b = {a.name};
    for i = 1:c
        r = round(rand(1)*length(b));
        disp(b(r))
        X = read(b(r));
    end
end

%Want this to return the Matrix of High and Lows for file
function X = read(file)
    mat = xlsread(file);
    High = mat(2);
    Low = mat(3);
    len = length(High);
    X = zeros(len, 2);
    for i = 1:len
        X(i, 1) = High(i);
        X(i, 2) = Low(i);
    end
end


