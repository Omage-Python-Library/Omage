"""
Omage Polish — Fine-tuning loop
"""


def polish(model, data, epochs=5, stop_loss=None, verbose=True):
    """
    Fine-tune a model
    
    Example:
        og.polish(my_model, my_data, epochs=5, stop_loss=0.01)
    """
    print(f"[Omage] Polish started — epochs: {epochs}")
    
    history = model.train(data, epochs=epochs, verbose=verbose)
    
    if stop_loss and history["loss"][-1] <= stop_loss:
        print(f"[Omage] Stopped early — loss {history['loss'][-1]:.4f} <= {stop_loss}")
    
    print("[Omage] Polish complete ✓")
    return history