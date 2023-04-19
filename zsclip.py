import torch
import numpy as np

def zero_shot_clip(model, tokenizer, images, true_class, prompts):
    pred_class = []
    pred_scores = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        prompts_token = tokenizer(prompts)
        text_features = model.encode_text(prompts_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        scores = (image_features @ text_features.T).softmax(dim=-1).detach().cpu().numpy()
        pred_class = scores.argmax(axis = 1)

    true_class = np.asarray(true_class)
    pred_class = np.asarray(pred_class)
    pred_scores = np.asarray(scores)

    acc = (true_class == pred_class).sum() / true_class.shape[0]
    print(f'Accuracy: {acc:.1%}')
    return true_class, pred_scores, acc