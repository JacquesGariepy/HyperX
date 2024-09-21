// components/SoundEffects.tsx
import { Howl } from 'howler';

export const playSound = (src: string) => {
  const sound = new Howl({ src });
  sound.play();
};
