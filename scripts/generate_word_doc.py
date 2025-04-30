from docx import Document


def create_doc(
    path, boot_S, p_S, boot_CI, p_CI, auc_S, auc_CI, motion, atlas, repl, mc
):
    doc = Document()
    doc.add_heading(
        "Parameter Sensitivity and Pilot Simulation of Synergy Metrics", level=1
    )

    doc.add_paragraph(
        f"ΔS={auc_S:.3f} (95% CI [{boot_S[0]:.3f}–{boot_S[1]:.3f}], "
        f"p_perm={p_S:.3f}); ΔCI={auc_CI:.3f} "
        f"(95% CI [{boot_CI[0]:.3f}–{boot_CI[1]:.3f}], p_perm={p_CI:.3f})."
    )
    doc.add_paragraph(
        f"Motion covariate: coef_awake={motion['coef_awake']:.3f}, "
        f"p={motion['p_awake']:.3f}."
    )
    for name, rough in atlas.items():
        doc.add_paragraph(
            f"Atlas {name}: roughness awake={rough['awake']:.3f}, "
            f"sedation={rough['sedation']:.3f}."
        )
    doc.add_paragraph(
        f"Replication ΔS={repl['delta_S']:.3f} "
        f"(95% CI [{repl['ci'][0]:.3f}–{repl['ci'][1]:.3f}]), "
        f"Cohen's d={repl['cohend']:.3f}."
    )
    for m, res in mc.items():
        doc.add_paragraph(
            f"Model {m}: ΔAUC={res['delta_auc']:.3f}, p={res['p_val']:.3f}."
        )
    doc.save(path)
