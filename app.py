    # 3. Noticias
    informe.append("## 3. Noticias recientes")
    tot_news = sum(len(v) for v in noticias_clas.values())
    if tot_news == 0:
        informe.append("- No se han podido obtener noticias recientes en esta fuente.")
    else:
        informe.append(f"- Se han encontrado **{tot_news} noticias**. Clasificación aproximada:")
        for categoria, lst in noticias_clas.items():
            if not lst:
                continue
            etiqueta = {
                "positivas": "✅ Positivas",
                "negativas": "❌ Negativas",
                "neutrales": "⚪ Neutrales",
            }[categoria]
            informe.append(f"  - {etiqueta}:")
            for n in lst[:5]:
                titulo = (n.get("title") or "Sin título").strip()
                fuente = (n.get("publisher") or "").strip()
                link = (n.get("link") or "").strip()

                texto_base = titulo
                if fuente:
                    texto_base += f" ({fuente})"

                if link:
                    informe.append(f"    - {texto_base} – {link}")
                else:
                    informe.append(f"    - {texto_base}")
        informe.append(
            "\nEstas noticias afectan al **sentimiento** de corto plazo, especialmente resultados, regulación, fusiones o cambios estratégicos."
        )
