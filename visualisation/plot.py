import matplotlib.pyplot as plt

class ObjectVisualizer:
    @staticmethod
    def plot_objects_row(mesh_objects, titles):
        num_objects = len(mesh_objects)
        fig, axes = plt.subplots(1, num_objects, figsize=(5 * num_objects, 5), subplot_kw={'projection': '3d'})
        if num_objects == 1:
            axes = [axes]

        for i, (mesh_object, title) in enumerate(zip(mesh_objects, titles)):
            ax = axes[i]
            if not mesh_object.is_empty:
                ax.plot_trisurf(
                    mesh_object.vertices[:, 0], mesh_object.vertices[:, 1], mesh_object.vertices[:, 2],
                    triangles=mesh_object.faces, alpha=0.8, edgecolor='k'
                )
                ax.set_title(title)
            else:
                print(f"Object {i} is empty or not loaded correctly.")

        plt.tight_layout()
        plt.show()
