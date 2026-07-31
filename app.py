import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

CLASS_NAMES = ['baluchari', 'chanderi', 'kanjeevaram', 'kosa',
               'kota_doria', 'muga', 'paithani', 'patola',
               'pochampally', 'uppada']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
REPO_ID     = 'SiddhantDCT/silk'
ALPHA       = 0.4   # tempering strength; lower = more conservative fusion

REGION_GUIDANCE = {
    'chanderi'   : 'Photograph the small gold buti motifs on the fabric body',
    'kanjeevaram': 'Photograph the wide border of the saree',
    'kosa'       : 'Photograph the fabric body texture up close (6-8 cm)',
    'paithani'   : 'Photograph the peacock motif in the pallu',
    'patola'     : 'Photograph the geometric ikat pattern on the body',
    'uppada'     : 'Photograph the floating woven motifs on the body',
    'pochampally': 'Photograph the geometric diamond ikat pattern',
    'muga'       : 'Photograph the fabric body to show the golden sheen',
    'baluchari'  : 'Photograph the pallu showing the narrative figure motifs',
    'kota_doria' : 'Photograph the fabric against light to show khat checks',
}

CLASS_DISPLAY = {
    'baluchari'  : 'Baluchari',
    'chanderi'   : 'Chanderi',
    'kanjeevaram': 'Kanjeevaram',
    'kosa'       : 'Kosa / Tussar',
    'kota_doria' : 'Kota Doria',
    'muga'       : 'Muga',
    'paithani'   : 'Paithani',
    'patola'     : 'Patola',
    'pochampally': 'Pochampally',
    'uppada'     : 'Uppada',
}

@st.cache_resource
def load_models():
    resnet_path = hf_hub_download(
        repo_id=REPO_ID, filename='resnet50_best.pth')
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(resnet.fc.in_features, NUM_CLASSES))
    resnet.load_state_dict(
        torch.load(resnet_path, map_location='cpu',
                   weights_only=True))
    resnet.eval()

    mobile_path = hf_hub_download(
        repo_id=REPO_ID, filename='mobilenet_v2_best.pth')
    mobile = models.mobilenet_v2(weights=None)
    mobile.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(mobile.classifier[1].in_features,
                  NUM_CLASSES))
    mobile.load_state_dict(
        torch.load(mobile_path, map_location='cpu',
                   weights_only=True))
    mobile.eval()

    return resnet, mobile


def predict_image(pil_image, model):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(
            model(tensor)[0], dim=0).numpy()
    return probs


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def tempered_bayesian_fusion(images, model, alpha=ALPHA,
                              diversity_aware=True, redundancy_penalty=1.5,
                              eps=1e-3):
    """
    Sequential tempered Bayesian fusion.
    Confidence builds as informative images are added, unlike plain
    softmax averaging.

    diversity_aware=True adds a redundancy penalty: if a new photo's
    probability vector looks similar to the average of photos already
    fused, it gets tempered harder (contributes less new evidence).
    This corrects for the fact that multiple photos of the same
    saree from similar angles are NOT independent evidence -- naive
    multiplication of probability vectors would overstate confidence.

    Returns:
        final_class, final_conf, individual (list of per-image results),
        posterior (final probability vector),
        trace (list of posterior vectors after each image, for plotting
               the confidence trajectory step by step)
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    posterior = np.ones(NUM_CLASSES) / NUM_CLASSES  # uniform prior
    individual = []
    seen_probs = []
    trace = []

    for img in images:
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor)[0], dim=0).numpy()

        pred_idx = probs.argmax()
        individual.append({'pred': CLASS_NAMES[pred_idx],
                            'conf': float(probs[pred_idx]) * 100})

        raw = np.clip(probs, eps, 1.0)

        if diversity_aware and seen_probs:
            avg_seen = np.mean(seen_probs, axis=0)
            sim = max(_cosine_sim(raw, avg_seen), 0.0)  # 0=novel, 1=redundant
            eff_alpha = min(alpha * (1 + redundancy_penalty * sim), 1.0)
        else:
            eff_alpha = alpha

        likelihood = np.power(raw, eff_alpha)
        posterior = posterior * likelihood
        posterior = posterior / posterior.sum()

        seen_probs.append(raw)
        trace.append(posterior.copy())

    final_idx   = posterior.argmax()
    final_class = CLASS_NAMES[final_idx]
    final_conf  = posterior[final_idx] * 100

    return final_class, final_conf, individual, posterior, trace


def plot_confidence_trajectory(trace, step_labels, final_class):
    """Small line chart showing how confidence in the final predicted
    class builds up as each image is fused in."""
    final_idx = CLASS_NAMES.index(final_class)
    confs = [float(p[final_idx]) * 100 for p in trace]
    st.line_chart(
        {"Confidence (%)": confs},
        x_label="Image added",
        y_label=f"Confidence in {CLASS_DISPLAY[final_class]} (%)")
    for label, c in zip(step_labels, confs):
        st.caption(f"After {label}: {c:.1f}%")


# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Silk Saree Classifier",
    layout="wide")

st.title("Indian Silk Saree Classification")
st.caption(
    "10 GI-Tagged Varieties  |  "
    "ResNet-50: 95.72%  |  "
    "MobileNetV2: 95.39%  |  "
    "NIT Silchar Research Project")
st.divider()

with st.spinner("Loading models..."):
    resnet_model, mobile_model = load_models()

# ── MODEL SELECTION ────────────────────────────────────────────
col_s1, col_s2 = st.columns(2)
with col_s1:
    model_choice = st.radio(
        "Select Model",
        ["ResNet-50 (95.72% accuracy — recommended)",
         "MobileNetV2 (95.39% accuracy — lightweight)"])
active_model = (resnet_model
                if "ResNet" in model_choice
                else mobile_model)
active_label = ("ResNet-50"
                if "ResNet" in model_choice
                else "MobileNetV2")

with col_s2:
    mode = st.radio(
        "Classification Mode",
        ["3-Step Protocol (recommended)",
         "Single image (quick)",
         "Multi-image ensemble"])

st.divider()

# ════════════════════════════════════════════════════════════════
# MODE 1 — 3-STEP PROTOCOL  (now uses Bayesian fusion, not averaging)
# ════════════════════════════════════════════════════════════════
if "3-Step" in mode:
    st.subheader("3-Step Classification Protocol")
    st.markdown(
        "Upload 3 photographs of the **same saree** following "
        "the steps below. The system fuses them sequentially using "
        "tempered Bayesian fusion — each image sharpens the belief "
        "rather than being simply averaged in.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Step 1")
        st.markdown("**Full saree photograph**")
        st.caption(
            "Stand back and photograph the entire saree. ")
        file1 = st.file_uploader(
            "Upload full view",
            type=['jpg','jpeg','png','webp'],
            key="step1")

    with col2:
        st.markdown("### Step 2")
        st.markdown("**Body texture close-up**")
        st.caption(
            "Photograph the fabric body 6-10 cm away. "
            "Show the texture, weave, or body motifs clearly.")
        file2 = st.file_uploader(
            "Upload body close-up",
            type=['jpg','jpeg','png','webp'],
            key="step2")

    with col3:
        st.markdown("### Step 3")
        st.markdown("**Pallu (decorated end)**")
        st.caption(
            "Photograph the decorated end of the saree. "
            "This shows the most distinctive motifs and patterns.")
        file3 = st.file_uploader(
            "Upload pallu",
            type=['jpg','jpeg','png','webp'],
            key="step3")

    st.markdown("---")

    files_uploaded = [f for f in [file1, file2, file3]
                      if f is not None]
    step_labels    = ["Full View", "Body Close-up", "Pallu"]
    uploaded_labels = [
        step_labels[i] for i, f in enumerate([file1, file2, file3])
        if f is not None]

    if len(files_uploaded) == 0:
        st.info(
            "Upload at least one image above to get started. "
            "For best results upload all three, in order.")

    elif len(files_uploaded) > 0:
        images = [Image.open(f).convert('RGB')
                  for f in files_uploaded]

        # ── FUSE (this is the Bayesian fusion call, in upload order) ──
        final_class, final_conf, individual, posterior, trace = \
            tempered_bayesian_fusion(images, active_model, alpha=ALPHA,
                                      diversity_aware=True)
        agreement = sum(r['pred'] == final_class for r in individual)

        # ── INDIVIDUAL STEP RESULTS ────────────────────────────
        st.subheader("Individual Step Results")
        ind_cols = st.columns(len(images))

        for i, (img, col, label, ind) in enumerate(
                zip(images, ind_cols, uploaded_labels, individual)):
            with col:
                st.image(img, use_column_width=True,
                         caption=f"Step: {label}")
                conf = ind['conf']
                pred_cls = ind['pred']

                if conf >= 90:
                    st.success(
                        f"{CLASS_DISPLAY[pred_cls]}\n{conf:.1f}%")
                elif conf >= 50:
                    st.warning(
                        f"{CLASS_DISPLAY[pred_cls]}\n{conf:.1f}%")
                else:
                    st.error(
                        f"{CLASS_DISPLAY[pred_cls]}\n{conf:.1f}%")

        st.divider()

        # ── FUSED (BAYESIAN) VERDICT ───────────────────────────
        st.subheader("Final Verdict (Bayesian Fusion)")

        vcol1, vcol2 = st.columns([1, 1])

        with vcol1:
            if final_conf >= 90:
                st.success(
                    f"### {CLASS_DISPLAY[final_class]}\n\n"
                    f"**Confidence: {final_conf:.1f}%**\n\n"
                    f"High confidence result.")
            elif final_conf >= 50:
                st.warning(
                    f"### {CLASS_DISPLAY[final_class]}\n\n"
                    f"**Confidence: {final_conf:.1f}%**\n\n"
                    f"{REGION_GUIDANCE[final_class]}")
            else:
                st.error(
                    f"### Not recognized\n\n"
                    f"Closest: {CLASS_DISPLAY[final_class]} "
                    f"({final_conf:.1f}%)\n\n"
                    f"This may not be one of the 10 trained "
                    f"GI silk varieties.")

            if len(images) > 1:
                st.metric(
                    "Step Agreement",
                    f"{agreement} / {len(images)}",
                    help="How many steps predicted the same class")
            st.caption(f"Model: {active_label}  |  Fusion: tempered Bayesian, alpha={ALPHA}")

            if len(images) > 1:
                st.markdown("**Per-step breakdown:**")
                for label, ind in zip(uploaded_labels, individual):
                    match = "✓" if ind['pred'] == final_class else "✗"
                    st.caption(
                        f"{match} {label}: "
                        f"{CLASS_DISPLAY[ind['pred']]} "
                        f"({ind['conf']:.1f}%)")

        with vcol2:
            st.markdown("**All class probabilities (fused):**")
            sorted_idx = np.argsort(posterior)[::-1]
            for idx in sorted_idx:
                cls  = CLASS_NAMES[idx]
                prob = float(posterior[idx] * 100)
                if idx == np.argmax(posterior):
                    st.markdown(
                        f"**{CLASS_DISPLAY[cls]}: {prob:.1f}%**")
                else:
                    st.write(
                        f"{CLASS_DISPLAY[cls]}: {prob:.1f}%")
                st.progress(int(prob))

        if len(images) > 1:
            st.markdown("**Confidence trajectory (as each step is fused in):**")
            plot_confidence_trajectory(trace, uploaded_labels, final_class)

        if len(files_uploaded) < 3:
            st.info(
                f"You uploaded {len(files_uploaded)}/3 images. "
                f"Upload all 3 steps for the most reliable result.")


# ════════════════════════════════════════════════════════════════
# MODE 2 — SINGLE IMAGE  (unchanged — no fusion needed for one image)
# ════════════════════════════════════════════════════════════════
elif "Single" in mode:
    st.subheader("Single Image Classification")
    file = st.file_uploader(
        "Upload one saree photograph",
        type=['jpg','jpeg','png','webp'])

    if file:
        img      = Image.open(file).convert('RGB')
        probs    = predict_image(img, active_model)
        pred_idx = probs.argmax()
        pred_cls = CLASS_NAMES[pred_idx]
        conf     = probs[pred_idx] * 100

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, use_column_width=True)
        with col2:
            if conf >= 90:
                st.success(
                    f"### {CLASS_DISPLAY[pred_cls]}\n\n"
                    f"**Confidence: {conf:.1f}%**")
            elif conf >= 40:
                st.warning(
                    f"### {CLASS_DISPLAY[pred_cls]}\n\n"
                    f"**Confidence: {conf:.1f}%**\n\n"
                    f"{REGION_GUIDANCE[pred_cls]}")
            else:
                st.error(
                    f"### Not recognized\n\n"
                    f"Confidence too low ({conf:.1f}%)\n\n"
                    f"This may not be a GI-tagged silk saree.")

            st.caption(f"Model: {active_label}")
            st.markdown("**All probabilities:**")
            for idx in np.argsort(probs)[::-1]:
                cls  = CLASS_NAMES[idx]
                prob = float(probs[idx] * 100)
                label = f"**{CLASS_DISPLAY[cls]}: {prob:.1f}%**" \
                        if idx == pred_idx \
                        else f"{CLASS_DISPLAY[cls]}: {prob:.1f}%"
                st.markdown(label)
                st.progress(int(prob))


# ════════════════════════════════════════════════════════════════
# MODE 3 — MULTI-IMAGE ENSEMBLE  (now uses Bayesian fusion; the
# previously-referenced ensemble_predict() didn't exist — replaced
# with a direct call to tempered_bayesian_fusion, in upload order)
# ════════════════════════════════════════════════════════════════
else:
    st.subheader("Multi-Image Ensemble")
    st.info(
        "Upload 5-10 photographs of the same saree from "
        "different angles and regions. Images are fused "
        "sequentially with tempered Bayesian fusion — order matters, "
        "so try to lead with your clearest, most distinctive shot.")

    files = st.file_uploader(
        "Upload 5-10 saree photographs",
        type=['jpg','jpeg','png','webp'],
        accept_multiple_files=True)

    if files and len(files) > 0:
        images = [Image.open(f).convert('RGB') for f in files]
        n = len(images)

        final_class, final_conf, individual, posterior, trace = \
            tempered_bayesian_fusion(images, active_model, alpha=ALPHA,
                                      diversity_aware=True)
        agreement = sum(r['pred'] == final_class for r in individual)

        cols = st.columns(min(n, 5))
        for i, (img, col) in enumerate(zip(images[:5], cols)):
            with col:
                ind   = individual[i]
                match = "OK" if ind['pred'] == final_class else "--"
                st.image(img, use_column_width=True,
                         caption=(f"Img {i+1} | "
                                  f"[{match}] "
                                  f"{CLASS_DISPLAY[ind['pred']]} "
                                  f"({ind['conf']:.0f}%)"))
        if n > 5:
            cols2 = st.columns(min(n-5, 5))
            for i, (img, col) in enumerate(zip(images[5:], cols2)):
                with col:
                    ind   = individual[i+5]
                    match = "OK" if ind['pred'] == final_class else "--"
                    st.image(img, use_column_width=True,
                             caption=(f"Img {i+6} | "
                                      f"[{match}] "
                                      f"{CLASS_DISPLAY[ind['pred']]} "
                                      f"({ind['conf']:.0f}%)"))

        st.divider()
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.subheader("Ensemble Result (Bayesian Fusion)")
            if final_conf >= 90:
                st.success(
                    f"### {CLASS_DISPLAY[final_class]}\n\n"
                    f"**Confidence: {final_conf:.1f}%**")
            elif final_conf >= 40:
                st.warning(
                    f"### {CLASS_DISPLAY[final_class]}\n\n"
                    f"**Confidence: {final_conf:.1f}%**\n\n"
                    f"{REGION_GUIDANCE[final_class]}")
            else:
                st.error(
                    f"Not recognized  |  {final_conf:.1f}%")
            st.metric("Image Agreement", f"{agreement} / {n}")
            st.caption(f"Model: {active_label}  |  Fusion: tempered Bayesian, alpha={ALPHA}")

        with col_r2:
            st.subheader("Class Probabilities (fused)")
            for idx in np.argsort(posterior)[::-1]:
                cls  = CLASS_NAMES[idx]
                prob = float(posterior[idx] * 100)
                if idx == np.argmax(posterior):
                    st.markdown(f"**{CLASS_DISPLAY[cls]}: {prob:.1f}%**")
                else:
                    st.write(f"{CLASS_DISPLAY[cls]}: {prob:.1f}%")
                st.progress(int(prob))

        st.markdown("**Confidence trajectory (as each image is fused in):**")
        plot_confidence_trajectory(
            trace, [f"Image {i+1}" for i in range(n)], final_class)

st.divider()
st.caption(
    "Silk Saree Classifier  |  "
    "ResNet-50 (95.72%) and MobileNetV2 (95.39%)  |  "
    "10 GI-Tagged Indian Silk Varieties  |  "
    "External validation: 96.0% accuracy on 50 unseen samples")
