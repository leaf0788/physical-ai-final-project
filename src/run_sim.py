from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from common import load_policy, heart_waypoints, policy_state, wrap_angle

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "output" / "heart_policy.pt"
VIDEO = ROOT / "output" / "PAI_final_heart_robot.mp4"
FIGURE = ROOT / "output" / "final_trajectory.png"


def simulate(max_steps=3600, dt=0.05):
    model = load_policy(MODEL)
    pts = heart_waypoints(n=90, scale=0.18)
    x, y = pts[0]
    theta = float(np.arctan2(pts[1,1]-pts[0,1], pts[1,0]-pts[0,0]))
    idx = 1
    hist = []
    for _ in range(max_steps):
        target = pts[idx]
        st = policy_state(x, y, theta, target)
        with torch.no_grad():
            v, w = model(torch.from_numpy(st).unsqueeze(0))[0].numpy()
        if st[2] < 0.22:
            v *= 0.65
        x += float(v) * np.cos(theta) * dt
        y += float(v) * np.sin(theta) * dt
        theta = float(wrap_angle(theta + float(w) * dt))
        hist.append((x, y, theta, float(v), float(w), idx))
        if st[2] < 0.11:
            idx += 1
            if idx >= len(pts):
                break
    return pts, np.array(hist, dtype=float)


def save_figure(pts, hist):
    plt.figure(figsize=(6,6))
    plt.plot(pts[:,0], pts[:,1], '--', label='target heart')
    plt.plot(hist[:,0], hist[:,1], label='AI policy trajectory')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('AI-controlled differential-drive robot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=180)
    plt.close()


def world_to_px(x, y, xmin, xmax, ymin, ymax, W, H):
    px = int(50 + (x-xmin)/(xmax-xmin) * (W-100))
    py = int(H-50 - (y-ymin)/(ymax-ymin) * (H-100))
    return px, py


def draw_frame(pts, hist, k, title, subtitle, W=960, H=720):
    img = np.full((H, W, 3), 245, dtype=np.uint8)
    xmin, xmax = pts[:,0].min()-1.0, pts[:,0].max()+1.0
    ymin, ymax = pts[:,1].min()-1.0, pts[:,1].max()+1.0

    # Target heart
    pxy = [world_to_px(x,y,xmin,xmax,ymin,ymax,W,H) for x,y in pts]
    for i in range(len(pxy)-1):
        if i % 2 == 0:
            cv2.line(img, pxy[i], pxy[i+1], (170,170,170), 2, cv2.LINE_AA)
    cv2.line(img, pxy[-1], pxy[0], (170,170,170), 2, cv2.LINE_AA)

    # Executed path
    if k > 1:
        trail = [world_to_px(x,y,xmin,xmax,ymin,ymax,W,H) for x,y in hist[:k+1,:2]]
        cv2.polylines(img, [np.array(trail, dtype=np.int32)], False, (40,40,40), 4, cv2.LINE_AA)

    x,y,th,v,w,idx = hist[k]
    cx,cy = world_to_px(x,y,xmin,xmax,ymin,ymax,W,H)
    # Robot triangle
    r = 18
    tri=[]
    for a,rr in [(0,r*1.4),(2.5,r),( -2.5,r)]:
        wx=x+(rr/70.0)*np.cos(th+a)
        wy=y+(rr/70.0)*np.sin(th+a)
        tri.append(world_to_px(wx,wy,xmin,xmax,ymin,ymax,W,H))
    cv2.fillConvexPoly(img, np.array(tri,dtype=np.int32), (20,20,20), cv2.LINE_AA)

    tx,ty = pxy[min(int(idx),len(pxy)-1)]
    cv2.drawMarker(img,(tx,ty),(80,80,80),cv2.MARKER_TILTED_CROSS,20,2,cv2.LINE_AA)

    cv2.putText(img, title, (40,45), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (20,20,20), 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (40,78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,40,40), 1, cv2.LINE_AA)
    cv2.putText(img, f'AI policy inference | waypoint {int(idx):02d}/{len(pts)-1} | v={v:.2f} w={w:.2f}',
                (40,H-45), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20,20,20), 1, cv2.LINE_AA)
    cv2.putText(img, 'Model: 2-layer MLP trained by imitation learning',
                (40,H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,60,60), 1, cv2.LINE_AA)
    return img


def save_video(pts, hist, fps=10):
    W,H = 960,720
    intro_n, motion_n, outro_n = 5*fps, 50*fps, 10*fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(VIDEO), fourcc, fps, (W,H))
    sample = np.linspace(0, len(hist)-1, motion_n).astype(int)
    for f in range(intro_n):
        vw.write(draw_frame(pts,hist,0,'Physical AI Final Project','Imitation-Learned Heart Drawing Robot',W,H))
    for i,k in enumerate(sample):
        vw.write(draw_frame(pts,hist,int(k),'AI Policy Inference in Progress','Neural policy outputs linear/angular velocity from relative target state',W,H))
    for f in range(outro_n):
        vw.write(draw_frame(pts,hist,len(hist)-1,'Task Completed','The learned policy followed the waypoint sequence and drew a heart',W,H))
    vw.release()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--no-video',action='store_true')
    args=p.parse_args()
    pts,hist=simulate()
    print(f'steps: {len(hist)}, final waypoint index: {int(hist[-1,5])}')
    save_figure(pts,hist)
    if not args.no_video:
        save_video(pts,hist)
        print(f'saved: {VIDEO}')
    print(f'saved: {FIGURE}')

if __name__=='__main__':
    main()
