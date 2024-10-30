function x = softlog(x)
    if x > 0
        x = log(x);
    else
        x = -Inf;
    end
end