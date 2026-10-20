# Anatomical meshes

BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.

- Source: [BodyParts3D 4.0, PART-OF polygon meshes](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in the [official archive terms](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html).
- Organ mapping: [official element list](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/partof_element_parts.txt).

`liver.vtp` combines the liver tissue elements FJ2816, FJ2818, FJ2819, FJ2820, FJ2821, FJ2822, FJ2409, FJ2823 and FJ2824. Their volumes were combined, small gaps closed, and the outer surface smoothed and simplified.

`gallbladder.vtp` derives from FJ2817. Its surface was subdivided and smoothed.

`stomach.vtp` derives from FJ2564 and `duodenum.vtp` from FJ2573. Their surfaces were Loop-subdivided and lightly smoothed, retaining their source proportions and natural pylorus junction.

`diaphragm.vtp` derives from diaphragm element FJ3131. Detached source artefacts were removed and a cranial dome region retained with a single crop along the atlas's anatomical axis; its cut edges were closed. It shares the liver and gallbladder source orientation and scale. This is a regional context model rather than the complete diaphragm. The application moves it with the adjacent liver during breathing.

`abdomen.vtp` derives from torso skin element FJ2810, cropped from the chest to the hips. Adjacent torso contours replace the arms at the upper crop. The surface was smoothed, enlarged and locally expanded to enclose the existing task anatomy throughout its breathing range and accommodate the instrument's permitted movement. It retains the source waist, abdominal and hip contours. An inset surface supplies the inner lining, and a small anterior patch accommodates the two established entry ports. The application closes this patch around two real port openings and displays a section in external views; the laparoscope and captures retain the full enclosure. The lining and port adaptation are project constructions rather than atlas peritoneum models.

The application rotates the liver, gallbladder, diaphragm and digestive group together around the insertion axis to align their shared source orientation with the torso and entry ports. Their proportions and the fundus target anchor are preserved. The torso is an adapted visual enclosure, not a patient-specific anatomical reconstruction.

`bowel.vtp` uses the small-intestinal elements FJ2574–FJ2598 and FJ2600–FJ2628, caecum FJ2599, ascending colon FJ2566, transverse colon FJ2572, descending colon FJ2567, small-intestinal mesentery FJ3396 and transverse mesocolon FJ3398. The small-intestinal volumes were combined and their outer surface smoothed; the colon was subdivided and smoothed. The mesenteries were smoothed after removing 121 zero-volume paired-triangle artefacts. The sigmoid colon and rectum are outside this regional model.

The stomach, duodenum and bowel share the liver's source orientation and scale, with one 20 mm anatomical-posterior translation for operative exposure. This preserves the digestive group's relative positions and junctions while clearing the existing instrument routes and laparoscope sightline. The application moves the group with the liver during breathing. The body lining and outer skin were locally enlarged to accommodate this layout across the simulator's scenarios.

All meshes were converted to VTP and adapted in scale, orientation and position for this simulator. These changes were made for the project and are not endorsed by the source authors.
