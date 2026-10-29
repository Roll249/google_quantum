import re, sys
import kickmix as km
def breakdown(c):
    s=c.flame_chart_svg(); tot=c.operation_statistics().num_toffoli_like
    items=re.findall(r'<rect x="([\d.]+)" y="(10|0)" width="([\d.]+)" height="10"[^>]*/>\s*<text[^>]*>([^<]+)</text>', s)
    agg={}
    for x,y,w,lab in items:
        if y=='0': agg[lab]=agg.get(lab,0)+float(w)*tot/1000
    return agg
if __name__=='__main__':
    c=km.Circuit.from_file(sys.argv[1]); print(breakdown(c))
