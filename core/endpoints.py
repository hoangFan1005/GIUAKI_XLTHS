"""Train-only noise calibration and final endpoint hysteresis; no LAB at inference."""
import math

def fit_noise_floor(records):
    """Đầu vào: features/labels TRAIN. Đầu ra: mean/std/Q95 và noise upper3sigma.

    Không nhận test; giữ raw normalized STE và tự tính thống kê bằng vòng lặp.
    Q95 dùng nearest-rank trên dữ liệu đã sắp, không gọi hàm percentile thư viện.
    """
    noise=[]
    for record in records:
        for value,label in zip(record['features']['ste_norm'],record['labels']):
            if label==0:noise.append(value)
    if not noise:
        raise ValueError('Endpoint calibration needs labeled training silence')
    total=0.
    for value in noise:total+=value
    mean=total/len(noise);squared=0.
    for value in noise:squared+=(value-mean)**2
    std=math.sqrt(squared/len(noise))
    ordered=sorted(noise)
    q95=ordered[max(0,math.ceil(.95*len(ordered))-1)]
    return dict(noise_mean=mean,noise_std=std,noise_q95=q95,
                noise_upper=mean+3*std,noise_count=len(noise),
                calibration='training silence only; population std; Q95 nearest rank',
                high_multiplier=1.5)

def endpoint_thresholds(base_threshold,noise,histogram=False):
    """Đầu vào: ngưỡng gốc đổi sang normalized STE, noise model train.

    Đầu ra: (low,high) tham gia detection. Với histogram, ngưỡng năng lượng
    xác nhận HIGH; LOW giữ phoneme yếu và được giới hạn bằng noise upper.
    TT1/TT3 giữ ngưỡng đã học, có sàn Q95 chống nền nhiễu ở bước LOW.
    """
    if not math.isfinite(base_threshold) or base_threshold<0:
        raise ValueError('Endpoint base threshold must be finite and nonnegative')
    upper=max(0.,noise['noise_upper'])
    low=max(noise.get('noise_q95',0.),min(base_threshold,upper) if histogram else base_threshold,1e-12)
    high=max(low*noise.get('high_multiplier',1.5),upper,base_threshold)
    return low,high

def hysteresis_regions(values,starts,ends,duration,low,high,min_silence=.2,min_speech=.1,seed_mask=None):
    """Đầu vào: STE/timing thực, LOW/HIGH, min durations theo giây; seed tùy chọn.

    Đầu ra: các FINAL (start,end) đã nối gap ngắn và lọc vùng ngắn, tăng dần.
    HIGH và seed xác nhận speech; LOW giữ speech. Xác nhận silence sau200ms
    nhưng trả END về support cuối trên LOW, không cộng thời gian chờ vào END.
    Giữ nhiều vùng nói nếu gap đủ dài; không biết tên WAV hay LAB.
    """
    if not(len(values)==len(starts)==len(ends)) or (seed_mask is not None and len(seed_mask)!=len(values)):
        raise ValueError('Endpoint values, timing and seed lengths differ')
    if not all(math.isfinite(v) for v in (duration,low,high,min_silence,min_speech)) or not(0<low<high and duration>=0 and min_silence>=0 and min_speech>=0):
        raise ValueError('Invalid endpoint thresholds or durations')
    previous=-1.
    for v,s,e in zip(values,starts,ends):
        if not all(math.isfinite(x) for x in (v,s,e)) or v<0 or not(0<=s<e<=duration+1e-12) or s<=previous:
            raise ValueError('Invalid endpoint frame support or STE')
        previous=s
    regions=[];active=False;first=None;last=None;low_start=None
    # LOW-only runs are candidates. They cannot create final speech without HIGH.
    for index,value in enumerate(values):
        # Check gap before reading a new HIGH/LOW frame, including exactly200ms.
        if active and starts[index]-ends[last]>=min_silence-1e-12:
            regions.append((starts[first],min(duration,ends[last])))
            active=False;low_start=None
        if not active:
            if value>=low:
                if low_start is None:low_start=index
                confirmed=value>=high and (seed_mask is None or seed_mask[index])
                if confirmed:active=True;first=low_start;last=index
            else:low_start=None
        elif value>=low:
            last=index
    # Flush speech touching the audio tail; remove short regions after gap merge.
    if active:regions.append((starts[first],min(duration,ends[last])))
    return [(s,e) for s,e in regions if e-s>=min_speech-1e-12]

def regions_to_mask(regions,starts,ends):
    """Đầu vào: FINAL regions và support khung. Đầu ra: mask phục vụ frame scores.

    Chỉ đánh dấu khung được chứa trọn trong FINAL region; không nới biên thêm.
    """
    result=[];cursor=0
    for s,e in zip(starts,ends):
        while cursor<len(regions) and s>=regions[cursor][1]-1e-12:cursor+=1
        result.append(int(cursor<len(regions) and s>=regions[cursor][0]-1e-12 and e<=regions[cursor][1]+1e-12))
    return result
